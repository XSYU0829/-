import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from numba import njit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@njit
def compute_fields(X, Y, U, a):
    """矢量化的复速度场和压力系数计算"""
    Z = X + 1j * Y
    mask = np.abs(Z) > a
    V = U * (1 - a ** 2 / Z ** 2)
    V_mag = np.abs(V)
    Cp = 1 - (V_mag / U) ** 2
    return np.where(mask, V, np.nan), np.where(mask, Cp, np.nan)


# 生成高密度网格（300x300 提升精度）
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# 初始参数
U = 1.0  # 流速
a = 1.0  # 半径
nu = 1.5e-5  # 运动粘度

# 创建带极坐标子图的界面
fig = plt.figure(figsize=(15, 9))
ax_flow = fig.add_subplot(121)
ax_polar = fig.add_subplot(122, polar=True)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.3)

# 添加控制组件
ax_a = plt.axes([0.1, 0.2, 0.8, 0.03])
slider_a = Slider(ax_a, '半径 a (m)', 0.5, 2.0, valinit=a, valstep=0.1)

ax_U = plt.axes([0.1, 0.15, 0.8, 0.03])
text_box_U = TextBox(ax_U, '流速 U (m/s)', initial=str(U))

ax_Re = plt.axes([0.1, 0.1, 0.8, 0.03])
text_box_Re = TextBox(ax_Re, '雷诺数 Re', initial='')


def save_to_csv(U, a):
    """保存圆柱表面压力系数到CSV"""
    theta = np.linspace(0, 2 * np.pi, 36)
    x_samples = a * np.cos(theta)
    y_samples = a * np.sin(theta)

    # 计算表面速度场和压力系数
    Z_samples = x_samples + 1j * y_samples
    V_samples = U * (1 - a ** 2 / Z_samples ** 2)
    Vx = V_samples.real
    Vy = -V_samples.imag  # 注意速度分量方向
    Cp = 1 - (np.abs(V_samples) / U) ** 2

    # 格式化保存数据（保留4位小数）
    data = np.column_stack((theta, x_samples, y_samples, Cp))
    np.savetxt(f'cylinder_surface_a{a}_U{U}.csv', data,
               fmt='%.4f', delimiter=',',
               header='theta(rad),x,y,Cp')


def update_plot(U, a):
    """更新可视化"""
    ax_flow.clear()
    ax_polar.clear()

    # 计算流场
    V, Cp = compute_fields(X, Y, U, a)
    Z = X + 1j * Y
    F = U * (Z + a ** 2 / Z)  # 复势

    # ====== 流场可视化 ======
    # 绘制等势线（黄色虚线）
    phi = F.real
    ax_flow.contour(X, Y, phi, levels=10, colors='yellow',
                    linestyles='dashed', linewidths=0.5)

    # 绘制流线（黑色实线）
    ax_flow.streamplot(X, Y, V.real, -V.imag, color='black',
                       linewidth=0.8, density=2, arrowsize=0.7)

    # 添加圆柱
    ax_flow.add_patch(plt.Circle((0, 0), a, color='red', alpha=0.3))

    # ====== 极坐标压力分布 ======
    theta = np.linspace(0, 2 * np.pi, 100)
    # 理论压力系数曲线
    Cp_theory = 1 - 4 * (np.sin(theta)) ** 2
    ax_polar.plot(theta, Cp_theory, 'b--', label='理论值')

    # 计算表面压力系数
    x_val = a * np.cos(theta)
    y_val = a * np.sin(theta)
    Z_val = x_val + 1j * y_val
    V_val = U * (1 - a ** 2 / Z_val ** 2)
    Cp_surface = 1 - (np.abs(V_val) / U) ** 2
    ax_polar.scatter(theta, Cp_surface, c='r', s=5, label='计算值')

    # 标注极值点
    max_idx = np.argmax(Cp_surface)
    ax_polar.plot(theta[max_idx], Cp_surface[max_idx], 'ro', markersize=8)
    ax_polar.text(theta[max_idx], Cp_surface[max_idx],
                  f'θ={np.degrees(theta[max_idx]):.1f}°\nCp={Cp_surface[max_idx]:.2f}',
                  ha='center', va='bottom')

    ax_polar.set_theta_zero_location('N')
    ax_polar.legend(loc='upper right')
    ax_polar.set_title("圆柱表面压力系数分布", pad=20)

    # ====== 物理验证标注 ======
    # 驻点验证（θ=0和π）
    Cp_stag = 1 - (np.abs(U * (1 - a ** 2 / (a ** 2))) / U) ** 2  # 理论应等于1
    ax_flow.text(0.05, 0.95,
                 f'驻点压力系数: {Cp_surface[0]:.4f} (理论值 1.000)\n'
                 f'90°处Cp: {Cp_surface[25]:.4f} (理论值 -3.000)',
                 transform=ax_flow.transAxes, backgroundcolor='white')

    ax_flow.set_title(f"圆柱绕流流场 (U={U} m/s, a={a} m)")
    ax_flow.axis('equal')

    # 保存最新数据
    save_to_csv(U, a)
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