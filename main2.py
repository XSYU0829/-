import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from numba import njit, prange

# 设置字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用于支持负号显示

@njit(parallel=True)
def compute_velocity(X, Y, U, a):
    """计算复速度场并屏蔽圆柱内部"""
    V = np.zeros_like(X, dtype=np.complex128)
    for i in prange(X.shape[0]):
        for j in prange(X.shape[1]):
            z = X[i, j] + 1j * Y[i, j]
            if np.abs(z) > a:
                V[i, j] = U * (1 - a**2 / z**2)
            else:
                V[i, j] = np.nan  # 屏蔽内部
    return V

# 生成网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 参数设置
U = 1.0  # 来流速度
a = 1.0  # 圆柱半径
nu = 1.5e-5  # 运动粘度

# 创建图形界面
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, bottom=0.3)

# 创建滑动条和输入框
ax_a = plt.axes([0.1, 0.2, 0.8, 0.03])
slider_a = Slider(ax_a, '半径 a (m)', 0.5, 2.0, valinit=a, valstep=0.1)

ax_U = plt.axes([0.1, 0.15, 0.8, 0.03])
text_box_U = TextBox(ax_U, '流速 U (m/s)', initial=str(U))

ax_Re = plt.axes([0.1, 0.1, 0.8, 0.03])
text_box_Re = TextBox(ax_Re, '雷诺数 Re', initial='')

# 计算并绘制初始流场
def update_plot(U, a):
    V = compute_velocity(X, Y, U, a)
    ax.clear()
    ax.streamplot(X, Y, V.real, -V.imag, color='blue', density=2)
    circle = plt.Circle((0, 0), a, color='red', alpha=0.3)
    ax.add_patch(circle)
    ax.set_title(f"圆柱绕流流场 (U={U} m/s, a={a} m)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    plt.draw()

# 滑动条回调函数
def update_a(val):
    global a
    a = val
    update_plot(U, a)
    # 计算雷诺数
    Re = 2 * U * a / nu
    text_box_Re.set_val(f"{Re:.2f}")

# 输入框回调函数
def update_U(val):
    global U
    try:
        U = float(val)
        if U < 1 or U > 10:
            raise ValueError
        update_plot(U, a)
        # 计算雷诺数
        Re = 2 * U * a / nu
        text_box_Re.set_val(f"{Re:.2f}")
    except:
        text_box_U.set_val(f"{U:.1f}")

# 驻点标注
def plot_stagnation_points(X, Y, V):
    # 计算速度模长
    V_mag = np.abs(V)
    # 找到速度模长最小的位置
    idx = np.unravel_index(np.argmin(V_mag, axis=None), V_mag.shape)
    # 标注驻点
    ax.plot(X[idx], Y[idx], 'ro', markersize=5)

# 绑定回调函数
slider_a.on_changed(update_a)
text_box_U.on_submit(update_U)

# 初始绘制
update_plot(U, a)

plt.show()