import numpy as np
import matplotlib.pyplot as plt

# 参数设置
U = 1.0      # 来流速度
a = 1.0      # 圆柱半径
x_range = np.linspace(-3, 3, 300)  # 计算域范围
y_range = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x_range, y_range)
Z = X + 1j*Y  # 创建复平面网格

def complex_potential(z, U, a):
    """计算合成复势Φ(z) = U(z + a²/z)"""
    return U * (z + (a**2)/z)

def velocity_field(z, U, a):
    """通过复势导数计算速度场：dΦ/dz = u - iv"""
    deriv = U * (1 - (a**2)/(z**2))
    return deriv.real, -deriv.imag  # 返回速度分量(u, v)

# 计算流函数（复势的虚部）
psi = complex_potential(Z, U, a).imag

# 计算速度场
u, v = velocity_field(Z, U, a)

# 创建画布
plt.figure(figsize=(10, 8), facecolor='white')

# 绘制流线图
plt.streamplot(X, Y, u, v,
               density=2, color='b',
               linewidth=1, arrowsize=1)

# 绘制圆柱表面
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(a*np.cos(theta), a*np.sin(theta),
        'r--', linewidth=2, label=f'Cylinder (a={a})')

# 添加特征标注
plt.title(f'Flow Around Cylinder (U={U}, a={a})')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()