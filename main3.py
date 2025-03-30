import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.pyplot as plt

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

# 计算速度场
V = compute_velocity(X, Y, U, a)

# 绘制流线图
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, V.real, -V.imag, color='blue', density=2)
circle = plt.Circle((0, 0), a, color='red', alpha=0.3)
plt.gca().add_patch(circle)
plt.title(f"圆柱绕流流场 (U={U} m/s, a={a} m)")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
def calculate_pressure_coefficient(theta, U, a):
    """计算圆柱表面压力系数"""
    z_samples = a * np.exp(1j * theta)
    V = U * (1 - (a**2)/(z_samples**2))
    V_mag = np.abs(V)
    Cp = 1 - (V_mag / U)**2
    return Cp

# 生成角度采样点
theta = np.linspace(0, 2*np.pi, 36)
Cp = calculate_pressure_coefficient(theta, U, a)

# 绘制极坐标压力分布
plt.figure(figsize=(8, 8))
ax = plt.subplot(polar=True)
ax.plot(theta, Cp, 'r-', lw=2)
ax.set_title("圆柱表面压力系数分布")
plt.show()