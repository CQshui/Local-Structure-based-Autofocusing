import os
import time

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from tqdm import tqdm
import torch
import torch.fft
from torchvision import transforms
import torch.nn.functional as F

# 计时器
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)  # 执行原函数并保留返回值
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result  # 返回原函数的执行结果
    return wrapper


class AngularBatch(object):
    def __init__(self, z_initial, input_pth=None, output_pth=None, lam=532e-9, pix=0.098e-6,
                 input_img=None, run=False, inline=False, get_filtered=False, fx_ref=None, fy_ref=None,
                 cut_size=[2025, 2148, 466, 180, 3, 180], compress=1):
        self._window_shape = None
        self._window_cache = None
        self.width = None
        self.height = None
        self.fft_img = None
        self.fft_width = None
        self.fft_height = None
        self.u0 = None
        self.cut_size = cut_size

        self.lam = lam
        self.pix = pix
        self.z_init = z_initial
        self.input_pth = input_pth
        self.output_pth = output_pth
        self.fig_num = 0
        self.img_names = 'test'
        self.img_name = 'test'
        self.input_img = input_img  # 是否直接输入torch格式图像
        self.run = run  # 是否继续当前图像
        self.inline = inline  # 是否是同轴
        self.get_filtered = get_filtered  # 是否获取频谱滤波后图像

        # 空间频率
        self.fx_ref = fx_ref
        self.fy_ref = fy_ref

        # 图像压缩比例
        self.compress = compress

    def hann2d(self, height, width, device):
        wy = torch.hann_window(height, periodic=False, device=device)
        wx = torch.hann_window(width, periodic=False, device=device)
        window = torch.outer(wy, wx)
        return window

    # @timer
    def start(self):
        if self.input_img is None:
            self.img_names = os.listdir(self.input_pth)
            self.img_names = [f for f in self.img_names if
                              any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]
            pbar = tqdm(total=len(self.img_names), desc='Reconstruct')
            for img_name in self.img_names:
                self.fig_num += 1
                self.img_name = img_name
                img_pth = os.path.join(self.input_pth, img_name)
                save_pth = os.path.join(self.output_pth, img_name)

                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)

                # 读取图片，生成灰度图
                img = Image.open(img_pth)
                self.width, self.height = img.size
                gray_image = img.convert("L")
                gray = np.asarray(gray_image)
                gray = gray.copy()  # 创建可写的副本
                gray_tensor = torch.from_numpy(gray).to(dtype=torch.float32).to('cuda')

                # 初始化 z
                current_z = self.z_init
                run = True

                while run:
                    # 执行重建
                    self.reconstruct(gray_tensor, current_z, save_pth)

                    # 询问用户是否输入新的 z
                    user_input = input(f"输入新的 z 值（浮点数）或按回车结束当前图像重建（图像：{self.img_name}）：")
                    try:
                        # 尝试将输入转换为浮点数
                        new_z = float(user_input)
                        current_z = new_z  # 更新当前 z
                    except ValueError:
                        # 输入不是浮点数，结束当前图像重建
                        run = False
                        print(f"结束当前图像重建：{self.img_name}")

                pbar.update(1)

        else:
            # 首先压缩图像
            self.pix = self.pix * self.compress

            # 判断输入是否为PIL图像
            if isinstance(self.input_img, Image.Image):
                img = self.input_img
                self.width, self.height = img.size
                gray_image = img.convert("L")  # 转为灰度图
                gray = np.asarray(gray_image)
                gray = gray.copy()  # 创建可写副本
                gray_tensor = torch.from_numpy(gray).to(dtype=torch.float32).cuda()
            else:  # 已经是torch tensor则直接使用
                gray_tensor = self.input_img
                # self.height, self.width = gray_tensor.shape

                # 处理复数张量的尺寸压缩
                original_height, original_width = gray_tensor.shape
                new_height = int(original_height / self.compress)
                new_width = int(original_width / self.compress)

                if gray_tensor.is_complex():
                    # 分别对实部和虚部进行插值
                    real_part = gray_tensor.real.unsqueeze(0).unsqueeze(0)
                    imag_part = gray_tensor.imag.unsqueeze(0).unsqueeze(0)

                    real_resized = F.interpolate(
                        real_part,
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    imag_resized = F.interpolate(
                        imag_part,
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    gray_tensor = torch.complex(real_resized, imag_resized)
                else:
                    # 如果是实数张量，直接插值
                    gray_tensor = F.interpolate(
                        gray_tensor.unsqueeze(0).unsqueeze(0),
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                self.height, self.width = gray_tensor.shape

            # 初始化 z
            current_z = self.z_init

            # 重建，result在gpu上，如要转为图像格式：torch.abs(result).cpu().numpy()
            result = self.reconstruct(gray_tensor, current_z)

            return result

    # @timer
    def reconstruct(self, gray_tensor, z, save_pth=None):
        # 裁剪频谱图，并移动到中心
        if not self.inline:
            # FFT变换生成频谱图
            self.u0 = torch.fft.fftshift(torch.fft.fft2(gray_tensor))
            # 超出灰度阈值，降幂
            u1 = torch.log(1 + torch.abs(self.u0))
            u1_cpu = u1.cpu().numpy()
            # fft_pth = os.path.join(save_pth, 'FFT.jpg')
            # plt.imsave(fft_pth, u1_cpu, cmap="gray")

            # 生成归一化后的频谱图（替代原文件读取步骤）
            u1_cpu_normalized = ((u1_cpu - u1_cpu.min()) / (u1_cpu.max() - u1_cpu.min()) * 255).astype(np.uint8)
            self.fft_img = u1_cpu_normalized

            self.cut()  # 不再需要文件路径参数
            self.move_to_center()
            self.u0 = torch.fft.ifft2(torch.fft.ifftshift(self.u0))
            u0_processed = self.u0
        else:
            self.u0 = gray_tensor
            u0_processed = self.u0

        # ========== 是否使用tukey窗 ==========
        tukey_choice = False
        if tukey_choice:
            # 检查缓存
            if self._window_cache is None or self._window_shape != (self.height, self.width):
                # 创建余弦窗口（GPU加速）
                edge_ratio = 0.1
                edge_h = int(self.height * edge_ratio)
                edge_w = int(self.width * edge_ratio)

                window = torch.ones(self.height, self.width, device='cuda', dtype=torch.float32)

                if edge_h > 0:
                    # 上下边缘
                    fade = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge_h, device='cuda')))
                    window[:edge_h, :] *= fade.unsqueeze(1)
                    fade_inv = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, edge_h, device='cuda')))
                    window[-edge_h:, :] *= fade_inv.unsqueeze(1)

                if edge_w > 0:
                    # 左右边缘
                    fade = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge_w, device='cuda')))
                    window[:, :edge_w] *= fade.unsqueeze(0)
                    fade_inv = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, edge_w, device='cuda')))
                    window[:, -edge_w:] *= fade_inv.unsqueeze(0)

                # 缓存窗口
                self._window_cache = window
                self._window_shape = (self.height, self.width)

            u0_processed = self.u0 * self._window_cache

        # 继续重建
        fx = torch.linspace(-1 / (2 * self.pix), 1 / (2 * self.pix), self.width, device='cuda')
        fy = torch.linspace(-1 / (2 * self.pix), 1 / (2 * self.pix), self.height, device='cuda')
        # FX, FY = torch.meshgrid(fx, fy)  # todo
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        temp = 1 - ((self.lam * FX) ** 2 + (self.lam * FY) ** 2)
        temp[temp < 0] = 0

        # 引入z
        g = torch.exp(1j * (2 * np.pi / self.lam) * z * torch.sqrt(temp))
        g[temp < 0] = 0
        g_shifted = torch.fft.fftshift(g)
        u1 = torch.fft.fft2(torch.fft.fftshift(u0_processed))
        u2 = u1 * g_shifted
        u3 = torch.fft.ifftshift(torch.fft.ifft2(u2))

        if self.get_filtered:
            return self.u0
        else:
            return u3

    def cut(self):  # 移除文件路径参数
        if len(self.cut_size) == 0 or self.fx_ref is not None:
            if self.fx_ref is not None:  # 离轴模式
                self.cut_size = self.calculate_cut_position()
                min_x, min_y, rectangle_width, rectangle_height = self.cut_size
                self.fft_img[0:min_y, min_x:self.fft_width] = 0
                self.fft_img[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
                self.fft_img[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
                self.fft_img[0:min_y + rectangle_height, 0:min_x] = 0

                self.fft_height, self.fft_width = self.fft_img.shape[:2]
                # 滤出最中心的高亮像素块
                _, binary_image = cv2.threshold(self.fft_img, 100, 255, cv2.THRESH_BINARY)
                binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
                contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                counter_x = []
                counter_y = []
                counter_w = []
                counter_h = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    counter_x.append(x)
                    counter_y.append(y)
                    counter_w.append(w)
                    counter_h.append(h)

                # 找到最大宽度的矩形并画出
                max_index = counter_w.index(max(counter_w))
                x_center = counter_x[max_index]
                y_center = counter_y[max_index]
                w_center = counter_w[max_index]
                h_center = counter_h[max_index]
                cv2.rectangle(self.fft_img, (x_center, y_center), (x_center + w_center, y_center + h_center),
                              (0, 255, 0), 8)

                delta_x = int(0.5 * w_center + x_center - 0.5 * self.fft_width)
                delta_y = int(0.5 * h_center + y_center - 0.5 * self.fft_height)
                self.cut_size.extend([delta_x, delta_y])
                # print(self.cut_size)

            else:
                self.fft_height, self.fft_width = self.fft_img.shape[:2]

                cv2.namedWindow('FFT', 0)
                cv2.setMouseCallback('FFT', self.on_mouse)
                cv2.imshow('FFT', self.fft_img)
                cv2.waitKey(0)

                # 滤出最中心的高亮像素块
                _, binary_image = cv2.threshold(self.fft_img, 180, 255, cv2.THRESH_BINARY)
                binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
                contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                counter_x = []
                counter_y = []
                counter_w = []
                counter_h = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    counter_x.append(x)
                    counter_y.append(y)
                    counter_w.append(w)
                    counter_h.append(h)

                # 找到最大宽度的矩形并画出
                max_index = counter_w.index(max(counter_w))
                x_center = counter_x[max_index]
                y_center = counter_y[max_index]
                w_center = counter_w[max_index]
                h_center = counter_h[max_index]
                cv2.rectangle(self.fft_img, (x_center, y_center), (x_center + w_center, y_center + h_center),
                              (0, 255, 0), 8)

                cv2.namedWindow('FFT with Bright Spots', 0)
                cv2.imshow("FFT with Bright Spots", self.fft_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                delta_x = int(0.5 * w_center + x_center - 0.5 * self.fft_width)
                delta_y = int(0.5 * h_center + y_center - 0.5 * self.fft_height)
                self.cut_size.extend([delta_x, delta_y])
                # print(self.cut_size)

        else:
            self.fft_height, self.fft_width = self.fft_img.shape[:2]

            min_x, min_y, rectangle_width, rectangle_height = self.cut_size
            self.fft_img[0:min_y, min_x:self.fft_width] = 0
            self.fft_img[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
            self.fft_img[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
            self.fft_img[0:min_y + rectangle_height, 0:min_x] = 0

            # 滤出最中心的高亮像素块
            _, binary_image = cv2.threshold(self.fft_img, 180, 255, cv2.THRESH_BINARY)
            binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
            contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            counter_x = []
            counter_y = []
            counter_w = []
            counter_h = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                counter_x.append(x)
                counter_y.append(y)
                counter_w.append(w)
                counter_h.append(h)

            # 找到最大宽度的矩形并画出
            max_index = counter_w.index(max(counter_w))
            x_center = counter_x[max_index]
            y_center = counter_y[max_index]
            w_center = counter_w[max_index]
            h_center = counter_h[max_index]
            cv2.rectangle(self.fft_img, (x_center, y_center), (x_center + w_center, y_center + h_center),
                          (0, 255, 0), 8)

            # cv2.namedWindow('FFT with Bright Spots', 0)
            # cv2.imshow("FFT with Bright Spots", self.fft_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            delta_x = int(0.5 * w_center + x_center - 0.5 * self.fft_width)
            delta_y = int(0.5 * h_center + y_center - 0.5 * self.fft_height)
            self.cut_size.extend([delta_x, delta_y])

    def on_mouse(self, event, x, y, flags, param):
        global point1, point2
        fft_copy = self.fft_img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(fft_copy, point1, 10, (0, 255, 0), 3)
            cv2.imshow('FFT', fft_copy)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(fft_copy, point1, (x, y), (255, 0, 0), 3)
            cv2.imshow('FFT', fft_copy)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(fft_copy, point1, point2, (0, 0, 255), 8)
            cv2.imshow('FFT', fft_copy)

            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            rectangle_width = abs(point1[0] - point2[0])
            rectangle_height = abs(point1[1] - point2[1])

            self.fft_img[0:min_y, min_x:self.fft_width] = 0
            self.fft_img[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
            self.fft_img[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
            self.fft_img[0:min_y + rectangle_height, 0:min_x] = 0
            cv2.imshow('FFT', self.fft_img)

            self.cut_size = [min_x, min_y, rectangle_width, rectangle_height]

    def move_to_center(self):
        try:
            min_x, min_y, rectangle_width, rectangle_height, delta_x, delta_y = self.cut_size
        except:
            min_x, min_y, rectangle_width, rectangle_height = self.cut_size

        # 每一张图用各自的delta_x和delta_y
        self.cut_size = self.cut_size[:4]

        self.u0[0:min_y, min_x:self.fft_width] = 0
        self.u0[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
        self.u0[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
        self.u0[0:min_y + rectangle_height, 0:min_x] = 0

        self.u0 = torch.roll(self.u0, -delta_x, dims=1)
        self.u0 = torch.roll(self.u0, -delta_y, dims=0)

    def calculate_cut_position(self):
        """精确计算单个频谱的截取区域"""
        width, height = self.width, self.height

        # 计算频率分辨率
        dfx = 1 / (width * self.pix)
        dfy = 1 / (height * self.pix)

        # 计算频域坐标偏移量
        dx = int(round(self.fx_ref / dfx))
        dy = int(round(self.fy_ref / dfy))

        # 计算峰值位置（FFT移位后坐标系）
        x_peak = (width // 2 + dx) % width
        y_peak = (height // 2 + dy) % height

        # 设置截取区域大小（固定值或比例）
        rect_w = width // 3  # 示例：图像宽度的1/8
        rect_h = height // 3  # 示例：图像高度的1/8

        # 计算截取区域
        min_x = max(0, x_peak - rect_w // 2)
        min_y = max(0, y_peak - rect_h // 2)
        rect_w = min(rect_w, width - min_x)  # 防止越界
        rect_h = min(rect_h, height - min_y)  # 防止越界

        return [min_x, min_y, rect_w, rect_h]


if __name__ == '__main__':
    image = AngularBatch(lam=532e-9,
                         pix=1e-6,
                         z_initial=0.0003,
                         input_pth=r'F:\dongjiayao\Data\VOC\tmp\input',
                         output_pth=r'F:\dongjiayao\Data\VOC\tmp')
    image.start()
