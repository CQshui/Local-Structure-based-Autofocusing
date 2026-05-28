#!/user/bin/python
# coding=utf-8
import sys
from pathlib import Path

from edge_prediction.model.RCF import RCF

# 将项目根目录添加到 sys.path
root_dir = Path(__file__).parent  # 根据实际情况调整层级
sys.path.append(str(root_dir))

import os, sys
from PIL import Image
import cv2
import argparse
import time
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from edge_prediction.dataloader.data_loader import BSDS_RCFLoader, prepare_image_torch_gpu
from edge_prediction.functions import *
from torch.utils.data import DataLoader, sampler
from edge_prediction.utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, isdir, isfile, splitext, split, abspath, dirname

from torch.cuda.amp import autocast, GradScaler

# 初始化梯度缩放器
scaler = GradScaler()

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float,
# parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--temperature', default=1, type=float, metavar='T',
                    help='temperature')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=21, type=int, metavar='E',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=4, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='S',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default=None
                    , type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/rcf')   # note
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default=r'F:\DongJiayao\Data\PASCAL')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
print('***', args.lr)

# 创建loss可视化目录
LOSS_PLOT_DIR = join(TMP_DIR, 'loss_plots')
if not isdir(LOSS_PLOT_DIR):
    os.makedirs(LOSS_PLOT_DIR)


class LossVisualizer:
    """Loss可视化类 - 只保存图像，不显示"""

    def __init__(self, plot_dir=LOSS_PLOT_DIR):
        self.plot_dir = plot_dir
        self.epoch_losses = []  # 每个epoch的平均loss
        self.batch_losses = []  # 所有batch的loss

    def update_epoch_loss(self, epoch, avg_loss):
        """更新epoch loss并保存图像"""
        self.epoch_losses.append((epoch, avg_loss))
        self._save_loss_plots(epoch)

    def update_batch_loss(self, epoch, batch_idx, batch_loss):
        """更新batch loss"""
        self.batch_losses.append((epoch, batch_idx, batch_loss))

    def _save_loss_plots(self, current_epoch):
        """保存loss图像"""
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制epoch loss
        if self.epoch_losses:
            epochs, losses = zip(*self.epoch_losses)
            ax1.plot(epochs, losses, 'b-', linewidth=2, label='Epoch Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss per Epoch')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 自动调整y轴范围
            if len(losses) > 1:
                loss_range = max(losses) - min(losses)
                if loss_range > 0:
                    ax1.set_ylim(min(losses) - 0.1 * loss_range,
                                 max(losses) + 0.1 * loss_range)

        # 绘制batch loss（最近几个epoch）
        if self.batch_losses:
            # 显示最近3个epoch的batch loss
            recent_epochs = max(1, current_epoch - 2)
            recent_batches = [b for b in self.batch_losses if b[0] >= recent_epochs]

            if recent_batches:
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                for epoch in range(recent_epochs, current_epoch + 1):
                    epoch_batches = [b for b in recent_batches if b[0] == epoch]
                    if epoch_batches:
                        batch_indices = [b[1] for b in epoch_batches]
                        batch_losses = [b[2] for b in epoch_batches]
                        color = colors[(epoch - recent_epochs) % len(colors)]
                        ax2.plot(batch_indices, batch_losses,
                                 color=color, alpha=0.7, linewidth=1,
                                 label=f'Epoch {epoch}')

                ax2.set_xlabel('Batch Index')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Batch Loss (Epochs {recent_epochs}-{current_epoch})')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

        plt.tight_layout()

        # 保存图像
        plot_path = join(self.plot_dir, f'loss_epoch_{current_epoch:03d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图表释放内存

        print(f"Loss plot saved: {plot_path}")

    def save_final_plot(self):
        """保存最终的loss图像"""
        if self.epoch_losses:
            self._save_loss_plots(len(self.epoch_losses) - 1)

def main():
    args.cuda = True
    # 初始化loss可视化
    loss_visualizer = LossVisualizer()

    # dataset
    train_dataset = BSDS_RCFLoader(root=args.dataset, split="train", resize=None, transform=False, get_pil=True,)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=1, drop_last=True, shuffle=True, pin_memory=True)

    model = RCF()

    model.cuda()
    model.apply(weights_init)
    # load_vgg16pretrain(model)     # NOTE: 除了RCF和VGGRegression，不加载预训练模型    todo

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)   # 允许加载部分模型，允许有模型缺失
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))

            freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']

            for name, param in model.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
                    print(f'冻结: {name}')
                else:
                    param.requires_grad = True
                    print(f'训练: {name}')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)     # todo
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('sgd', args.lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []
    for epoch in range(args.start_epoch, args.maxepoch):
        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch,
            save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))

        log.flush()  # write log

        # Save checkpoint
        save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)
        scheduler.step()  # will adjust learning rate

        # 更新loss可视化
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss
        loss_visualizer.update_epoch_loss(epoch, tr_avg_loss)

    # 保存最终loss图像
    loss_visualizer.save_final_plot()

def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()

    epoch_loss = []
    counter = 0

    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()

        outputs = model(image)
        loss = torch.zeros(1).cuda()

        # for o in outputs:
        for j in range(len(outputs)):
            o, o_defocus = outputs[j], None
            loss = loss + cross_entropy_loss_RCF(o, label)

        counter += 1
        loss = loss / args.itersize
        loss.backward()

        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            label_out = torch.eq(label, 1).float()
            outputs.append(label_out)
            outputs.append(1 - image)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(1 - all_results, join(save_dir, "iter-%d.jpg" % i))

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    main()
