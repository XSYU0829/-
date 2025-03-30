# %% 导入库
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import pandas as pd
from ipywidgets import interact, FloatSlider
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# 设置字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用于支持负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示


# %% 复势函数及速度场计算
@njit(parallel=True)
def complex_velocity(X, Y, U, a):
    """
    计算复速度场 V(z) = U*(1 - a²/z²)
    返回速度分量 (u, v)
    """
    Z = X + 1j * Y
    mask = np.abs(Z) > a  # 屏蔽圆柱内部
    V = np.zeros_like(Z, dtype=np.complex128)

    for i in prange(Z.shape[0]):
        for j in prange(Z.shape[1]):
            if mask[i, j]:
                V[i, j] = U * (1 - (a ** 2) / (Z[i, j] ** 2))
            else:
                V[i, j] = 0j
    return V.real, -V.imag  # 注意虚部符号


# %% 流场可视化类
class FlowVisualizer:
    def __init__(self, a=1.0, U=1.0, extent=(-3, 3, -2, 2)):
        self.a = a
        self.U = U
        self.extent = extent
        self.grid_resolution = 0.05
        self._init_grid()
        self.nu = 1.5e-5  # 空气运动粘度

    def _init_grid(self):
        """初始化计算网格"""
        x = np.arange(self.extent[0], self.extent[1], self.grid_resolution)
        y = np.arange(self.extent[2], self.extent[3], self.grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)

    @property
    def Re(self):
        """雷诺数计算"""
        return 2 * self.U * self.a / self.nu

    def update_flow(self, a, U):
        """更新流动参数"""
        self.a = a
        self.U = U
        self._init_grid()

    def plot_flow(self):
        """绘制流场图"""
        u, v = complex_velocity(self.X, self.Y, self.U, self.a)
        speed = np.sqrt(u ** 2 + v ** 2)

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        # 流场主图
        ax = fig.add_subplot(gs[0])
        ax.streamplot(self.X, self.Y, u, v, color='white', linewidth=1, density=2)

        # 等势线
        phi = self.U * (self.X + (self.a ** 2 * self.X) / (self.X ** 2 + self.Y ** 2))
        cs = ax.contour(self.X, self.Y, phi, levels=15, colors='yellow', linestyles='dashed')

        # 圆柱表面
        circle = plt.Circle((0, 0), self.a, color='red', alpha=0.3)
        ax.add_patch(circle)

        # 驻点标注
        ax.plot([-self.a, self.a], [0, 0], 'ro', markersize=4)

        ax.set_xlim(self.extent[:2])
        ax.set_ylim(self.extent[2:])
        ax.set_aspect('equal')
        ax.set_title(f"圆柱绕流可视化 (Re={self.Re:.1f})")

        # 压力分布子图
        ax_polar = fig.add_subplot(gs[1], polar=True)
        theta = np.linspace(0, 2 * np.pi, 36)
        z_samples = self.a * np.exp(1j * theta)
        V = self.U * (1 - (self.a ** 2) / (z_samples ** 2))
        Cp = 1 - (np.abs(V) / self.U) ** 2
        ax_polar.plot(theta, Cp, 'r-', lw=2)
        ax_polar.set_title("表面压力系数")

        plt.tight_layout()
        plt.show()


# %% 交互界面
visualizer = FlowVisualizer()


@interact(
    a=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0, description='半径 (m)'),
    U=FloatSlider(min=1.0, max=10.0, step=0.5, value=1.0, description='流速 (m/s)')
)
def update_visualization(a, U):
    visualizer.update_flow(a, U)
    visualizer.plot_flow()


# %% 气动参数计算类
class AerodynamicAnalyzer:
    def __init__(self, U, a):
        self.U = U
        self.a = a

    def calculate_pressure(self, theta):
        """计算指定角度的压力系数"""
        z = self.a * np.exp(1j * theta)
        V = self.U * (1 - (self.a ** 2) / (z ** 2))
        return 1 - (np.abs(V) / self.U) ** 2

    def export_report(self, filename):
        """导出气动数据报告"""
        theta = np.linspace(0, 2 * np.pi, 36)
        z_samples = self.a * np.exp(1j * theta)
        V = self.U * (1 - (self.a ** 2) / (z_samples ** 2))
        Cp = 1 - (np.abs(V) / self.U) ** 2

        df = pd.DataFrame({
            'theta_deg': np.degrees(theta),
            'Cp': Cp,
            'V_real': V.real,
            'V_imag': V.imag
        })
        df.to_csv(filename, index=False, float_format='%.4f')


# %% 示例用法
if __name__ == "__main__":
    # 气动分析示例
    analyzer = AerodynamicAnalyzer(U=2.0, a=1.5)
    analyzer.export_report('aerodynamic_report.csv')

    # 生成理论压力分布图
    theta = np.linspace(0, 2 * np.pi, 100)
    Cp_theory = 1 - 4 * np.sin(theta) ** 2

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar=True)
    ax.plot(theta, Cp_theory, 'b--', label='理论解')
    ax.plot(theta, analyzer.calculate_pressure(theta), 'r-', label='计算值')
    ax.set_title("压力系数对比 (a=1.5m, U=2m/s)")
    ax.legend()
    plt.show()