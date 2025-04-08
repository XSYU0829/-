import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from numba import njit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@njit
def compute_velocity(X, Y, U, a):
    """矢量化的复速度场计算"""
    Z = X + 1j * Y
    mask = np.abs(Z) > a
    V = U * (1 - a ** 2 / Z ** 2)
    return np.where(mask, V, np.nan)


# 生成高密度网格（300x300 提升精度）
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# 初始参数
U = 1.0  # 流速
a = 1.0  # 半径
nu = 1.5e-5  # 运动粘度

# 创建界面
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, bottom=0.3)

# 添加控制组件
ax_a = plt.axes([0.1, 0.2, 0.8, 0.03])
slider_a = Slider(ax_a, '半径 a (m)', 0.5, 2.0, valinit=a, valstep=0.1)

ax_U = plt.axes([0.1, 0.15, 0.8, 0.03])
text_box_U = TextBox(ax_U, '流速 U (m/s)', initial=str(U))

ax_Re = plt.axes([0.1, 0.1, 0.8, 0.03])
text_box_Re = TextBox(ax_Re, '雷诺数 Re', initial='')


def update_plot(U, a):
    """更新可视化"""
    ax.clear()

    # 计算流场
    V = compute_velocity(X, Y, U, a)
    Z = X + 1j * Y
    F = U * (Z + a ** 2 / Z)  # 复势

    # 绘制等势线（黄色虚线）
    phi = F.real
    ax.contour(X, Y, phi, levels=10, colors='yellow',
               linestyles='dashed', linewidths=0.5)

    # 绘制流线（白色实线）
    ax.streamplot(X, Y, V.real, -V.imag, color='white',
                  linewidth=0.8, density=2, arrowsize=0.7)

    # 添加圆柱
    ax.add_patch(plt.Circle((0, 0), a, color='red', alpha=0.3))

    # 边界层位移厚度修正示例
    def effective_radius(a, Re):
        delta = 5 * a / np.sqrt(Re)
        return a + 0.3 * delta  # 经验修正系数

    # 标注驻点（理论位置）
    ax.plot(a, 0, 'ro', markersize=5, zorder=3)
    ax.plot(-a, 0, 'ro', markersize=5, zorder=3)

    # 物理验证
    theta = np.linspace(0, 2 * np.pi, 10)
    x_val = a * np.cos(theta)
    y_val = a * np.sin(theta)
    F_val = U * (x_val + 1j * y_val + a ** 2 / (x_val + 1j * y_val))
    std_psi = np.std(F_val.imag)
    ax.text(0.05, 0.95, f'流函数标准差: {std_psi:.2e} (<0.01Ua)',
            transform=ax.transAxes, backgroundcolor='white')

    ax.set_title(f"圆柱绕流流场 (U={U} m/s, a={a} m)")
    ax.axis('equal')
    plt.draw()


def update_a(val):
    global a
    a = val
    update_plot(U, a)
    text_box_Re.set_val(f"{2 * U * a / nu:.2f}")


def update_U(val):
    global U
    try:
        U = np.clip(float(val), 1.0, 10.0)
        update_plot(U, a)
        text_box_Re.set_val(f"{2 * U * a / nu:.2f}")
    except:
        text_box_U.set_val(f"{U:.1f}")


# 绑定事件
slider_a.on_changed(update_a)
text_box_U.on_submit(update_U)

# 初始绘制
update_plot(U, a)
plt.show()