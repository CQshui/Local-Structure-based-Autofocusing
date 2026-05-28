import time

import numpy as np
import sympy as sp

# 计时器
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)  # 执行原函数并保留返回值
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result  # 返回原函数的执行结果
    return wrapper


def brent_method(func, lb=-500, ub=500, tol=1e-8, iter_num=100):
    a, c = lb, ub
    b = (a + c) / 2.0

    GOLD = (3 - np.sqrt(5)) / 2  # 黄金分割常数

    x = b
    w = b
    v = b

    fw = func(w)
    fx = fw
    fv = fw

    e = 0
    d = 0
    for i in range(iter_num):
        xm = (a + c) / 2.0
        tol1 = tol * abs(x) + np.finfo(float).eps
        tol2 = 2 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (c - a)):
            return x

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d

            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (c - x):
                if x >= xm:
                    e = a - x
                else:
                    e = c - x
                d = GOLD * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or c - u < tol2:
                    d = np.sign(tol1) * (xm - x)
        else:
            if x >= xm:
                e = a - x
            else:
                e = c - x
            d = GOLD * e

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + np.sign(tol1) * d

        fu = func(u)

        # print('score:', -fu, '    z:', u)     # note 打印结果
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                c = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return x


# @timer
def multi_start_brent(func, lb=-500, ub=500, num_starts=3, method='uniform', tol=1e-8, iter_num=100, window_size=1e-5):
    best_x = None
    best_fx = float('inf')

    if method == 'uniform':
        step = (ub - lb) / num_starts
        for i in range(num_starts):
            a = lb + i * step
            c = a + step
            x = brent_method(func, a, c, tol, iter_num)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x
    elif method == 'random':
        for _ in range(num_starts):
            mid = np.random.uniform(lb, ub)
            a = max(lb, mid - window_size / 2)
            c = min(ub, mid + window_size / 2)
            x = brent_method(func, a, c, tol, iter_num)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x
    else:
        raise ValueError("Invalid method. Choose 'uniform' or 'random'.")

    return best_x


if __name__ == '__main__':
    alpha = sp.symbols('alpha')
    func = sp.lambdify(alpha, (alpha - 2) ** 2)

    # 使用多起点策略，这里选择随机方法，20次起点，窗口大小为100
    x_min = multi_start_brent(func, lb=-500, ub=500, num_starts=20, method='random', window_size=100)
    print(f'min x = {x_min}，f(x) = {func(x_min)}')