import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import simpson
from numba import njit
from matplotlib import rcParams

# 增强版字体配置（解决数学符号缺失问题）
rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei'],  # 优先使用雅黑字体
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'STIXGeneral',  # 显式设置数学字体
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'axes.formatter.use_mathtext': True  # 强制使用数学文本渲染
})

@njit
def compute_aerodynamics(theta, a, U, rho, Gamma):
    """含环量的气动力计算"""
    # 表面速度分布
    V_theta = -2 * U * np.sin(theta) + Gamma / (2 * np.pi * a)
    # 压力系数 (伯努利方程)
    Cp = 1 - (V_theta / U) ** 2  # 已修正此处特殊字符
    # 压力分布
    p = 0.5 * rho * U  ** 2 * Cp
    # 笛卡尔坐标分量
    dx = -np.sin(theta)
    dy = np.cos(theta)
    return p * dx, p * dy, Cp


class AerodynamicsAnalyzer:
    def __init__(self):
        self.fig = plt.figure(figsize=(18, 10))
        self._init_parameters()
        self._create_widgets()
        self._setup_plots()
        self.update_plots()

    def _init_parameters(self):
        """初始化物理参数"""
        self.U = 5.0  # 流速 (m/s)
        self.a = 1.0  # 圆柱半径 (m)
        self.rho = 1.2  # 空气密度 (kg/m³)
        self.Gamma = 0.0  # 初始环量 (m²/s)

    def _create_widgets(self):
        """创建交互控件"""
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.3)
        ax_G = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_G = Slider(ax_G, '环量Γ (m²/s)', -10, 10, valinit=0)
        self.slider_G.on_changed(self._update_gamma)

        ax_U = plt.axes([0.2, 0.2, 0.6, 0.03])
        self.slider_U = Slider(ax_U, '流速U (m/s)', 1, 20, valinit=5)
        self.slider_U.on_changed(self._update_velocity)

        self.report_btn = Button(plt.axes([0.8, 0.15, 0.1, 0.05]), '导出报告')
        self.report_btn.on_clicked(self._export_report)

    def _setup_plots(self):
        """配置科学可视化图表"""
        # 流场可视化
        self.ax_flow = self.fig.add_subplot(231)
        self.ax_flow.set_title("绕流流线分布")

        # 极坐标压力分布
        self.ax_polar = self.fig.add_subplot(232, polar=True)
        self.ax_polar.set_theta_zero_location('N')
        self.ax_polar.set_title("表面压力系数分布")

        # 气动力分解
        self.ax_force = self.fig.add_subplot(233)
        self.ax_force.set_title("气动力分量随环量变化")

        # 参数敏感性分析
        self.ax_sense = self.fig.add_subplot(212)
        self.ax_sense.set_title("升力系数敏感性分析")

    def _update_gamma(self, val):
        self.Gamma = val
        self.update_plots()

    def _update_velocity(self, val):
        self.U = val
        self.update_plots()

    def _export_report(self, event):
        """生成PDF分析报告"""
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('气动力分析报告.pdf') as pdf:
            plt.figure(figsize=(8.27, 11.69))
            plt.text(0.5, 0.7, '气动力分析报告\n环量修正模型', ha='center')
            pdf.savefig()
            plt.close()

            fig = self._create_report_figure()
            pdf.savefig(fig)
            plt.close(fig)

    def update_plots(self):
        """综合可视化更新"""
        self._update_flow_field()
        self._update_pressure_dist()
        self._update_force_components()
        self._update_sensitivity()
        plt.draw()

    def _update_flow_field(self):
        """含环量的流场更新"""
        self.ax_flow.clear()

        x = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, x)
        Z = X + 1j * Y
        F = self.U * (Z + self.a  ** 2 / Z) + 1j * self.Gamma / (2 * np.pi) * np.log(Z / self.a)

        psi = F.imag
        self.ax_flow.contour(X, Y, psi, levels=20, colors='blue', linewidths=0.8)
        self.ax_flow.add_patch(plt.Circle((0, 0), self.a, color='r', alpha=0.3))
        self.ax_flow.set_aspect('equal')

    def _update_pressure_dist(self):
        """压力分布对比更新"""
        self.ax_polar.clear()
        theta = np.linspace(0, 2 * np.pi, 100)

        Cp_theory = 1 - 4 * np.sin(theta)  ** 2
        self.ax_polar.plot(theta, Cp_theory, 'b--', label='无环量理论解')

        _, _, Cp = compute_aerodynamics(theta, self.a, self.U, self.rho, self.Gamma)
        self.ax_polar.plot(theta, Cp, 'r-', lw=2, label=f'Γ={self.Gamma:.1f}')
        self.ax_polar.legend()

    def _update_force_components(self):
        """气动力分解计算"""
        self.ax_force.clear()
        Gamma_range = np.linspace(-10, 10, 50)
        Lifts, Drags = [], []

        theta = np.linspace(0, 2 * np.pi, 100)  # 明确定义theta参数

        for G in Gamma_range:
            fx, fy, _ = compute_aerodynamics(theta, self.a, self.U, self.rho, G)
            L = simpson(fy, theta) * self.a
            D = simpson(fx, theta) * self.a
            Lifts.append(L)
            Drags.append(D)

        theory_lift = self.rho * self.U * Gamma_range
        self.ax_force.plot(Gamma_range, Lifts, 'ro', label='压力积分法')
        self.ax_force.plot(Gamma_range, theory_lift, 'b--', label='K-J定理')
        self.ax_force.set_xlabel('环量Γ [m²/s]')
        self.ax_force.set_ylabel('升力 [N/m]')
        self.ax_force.legend()

        self.ax_force.text(0.5, 0.9, f"计算阻力: {np.mean(Drags):.2e} N/m\n(理想流体零阻力)",
                           transform=self.ax_force.transAxes)

    def _update_sensitivity(self):
        """参数敏感性分析"""
        self.ax_sense.clear()
        U_values = np.linspace(1, 20, 20)
        lifts = [self.rho * U * self.Gamma for U in U_values]
        self.ax_sense.plot(U_values, lifts, 'g-', lw=2)
        self.ax_sense.set_xlabel('流速 [m/s]')
        self.ax_sense.set_ylabel('理论升力 [N/m]')

    def _create_report_figure(self):
        """生成报告用对比图表"""
        fig = plt.figure(figsize=(11, 8))

        ax1 = fig.add_subplot(211)
        theta = np.linspace(0, 2 * np.pi, 100)
        for G in [-8, 0, 8]:
            _, _, Cp = compute_aerodynamics(theta, self.a, self.U, self.rho, G)
            ax1.plot(np.degrees(theta), Cp, label=f'Γ={G}')
        ax1.set_ylabel('Cp')
        ax1.legend()

        ax2 = fig.add_subplot(212)
        Gamma = np.linspace(-10, 10, 50)
        lift = self.rho * self.U * Gamma
        ax2.plot(Gamma, lift, 'r-')
        ax2.set_xlabel('环量Γ [m²/s]')
        ax2.set_ylabel('升力 [N/m]')

        plt.figtext(0.1, 0.05,
                    "模型局限性:\n"
                    "1. 忽略粘性效应导致的流动分离\n"
                    "2. 稳态假设无法模拟涡脱落现象\n"
                    "3. 实际后驻点位置可能失效",
                    bbox={'facecolor': 'lightgray'})

        return fig


# 运行分析系统
analyzer = AerodynamicsAnalyzer()
plt.show()