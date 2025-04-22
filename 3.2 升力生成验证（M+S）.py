import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font
import platform


# ====================== 中文字体配置增强版 ======================
def configure_chinese_font():
    """跨平台字体配置（支持Windows/Linux/macOS）"""
    root = tk.Tk()
    root.withdraw()  # 隐藏Tk窗口

    # 动态字体加载
    font_table = {
        'Windows': 'Microsoft YaHei',
        'Linux': 'WenQuanYi Micro Hei',
        'Darwin': 'PingFang SC'
    }
    system = platform.system()

    # 配置全局字体设置
    plt.rcParams.update({
        'font.sans-serif': [font_table.get(system, 'Arial')],
        'axes.unicode_minus': False  # 解决负号显示问题
    })
    root.destroy()  # 销毁Tk对象


# ====================== 任务3.2：升力验证（公式精确实现） ======================
class EnhancedLiftValidator:
    def __init__(self, U=1.0, a=1.0, rho=1.2):
        """
        参数初始化
        U : 来流速度 (m/s)
        a : 圆柱半径 (m)
        rho : 流体密度 (kg/m³)
        """
        self.U = U
        self.a = a
        self.rho = rho

    def theoretical_lift(self, Gamma):
        """库塔-茹科夫斯基定理（精确公式实现）"""
        return self.rho * self.U * Gamma  # L = ρUΓ

    def pressure_coefficient(self, Gamma, theta):
        """压力系数计算（公式修正）"""
        V_theta = (Gamma / (2 * np.pi * self.a)) + 2 * self.U * np.sin(theta)
        return 1 - (V_theta / self.U) ** 2  # 修正压力系数公式

    def numerical_integration(self, Gamma, tolerance=0.1):
        """
        自适应辛普森积分（修正积分项系数）
        参数：
        tolerance : 允许的数值误差百分比 (默认0.1%)
        """
        n_points = 1000  # 初始采样点
        L_theory = self.theoretical_lift(Gamma)

        # 积分过程优化
        for _ in range(5):  # 最大5次迭代
            theta = np.linspace(0, 2 * np.pi, n_points)
            Cp = self.pressure_coefficient(Gamma, theta)

            # 修正后的积分项公式
            integrand = (-0.5 * self.rho * self.U ** 2 * Cp *
                         self.a * np.sin(theta))

            L_num = simpson(integrand, theta)
            error = abs((L_num - L_theory) / L_theory) * 100

            if error < tolerance:
                return L_num, error
            n_points *= 2  # 采样点倍增

        return L_num, error

    def flow_visualization(self, Gamma):
        """流场可视化（修正驻点计算和复速度场）"""
        x = y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # 复速度场计算（势流理论精确实现）
        with np.errstate(divide='ignore', invalid='ignore'):
            W = (self.U * (1 - (self.a ** 2) / (Z ** 2)) +
                 1j * Gamma / (2 * np.pi * Z))  # 点涡项修正

        # 可视化参数优化
        plt.figure(figsize=(10, 8))
        speed = np.sqrt(W.real ** 2 + W.imag ** 2)
        strm = plt.streamplot(X, Y, W.real, -W.imag,
                              color=self.pressure_coefficient(Gamma, np.arctan2(Y, X)),
                              cmap='coolwarm', density=2, linewidth=1,
                              arrowsize=1.2)

        # 颜色条设置
        cb = plt.colorbar(strm.lines, label='压力系数 Cp')
        cb.set_ticks(np.linspace(-3, 1, 9))

        # 圆柱标注
        circle = plt.Circle((0, 0), self.a, color='k', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

        # 理论驻点计算（修正角度处理）
        with np.errstate(invalid='ignore'):
            theta_stag = np.arcsin(-Gamma / (4 * np.pi * self.U * self.a))
        valid_mask = ~np.isnan(theta_stag)

        if np.any(valid_mask):
            # 主驻点
            x_stag = self.a * np.cos(theta_stag)
            y_stag = self.a * np.sin(theta_stag)
            # 对称驻点
            x_stag2 = self.a * np.cos(np.pi - theta_stag)
            y_stag2 = self.a * np.sin(np.pi - theta_stag)

            plt.scatter([x_stag, x_stag2], [y_stag, y_stag2],
                        c='r', s=80, edgecolors='w',
                        zorder=3, label='驻点')

        plt.title(f"环量Γ={Gamma}时的流场模式\n理论升力={self.theoretical_lift(Gamma):.2f}N/m",
                  fontsize=12, pad=15)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()


# ====================== 主程序（优化验证） ======================
if __name__ == "__main__":
    configure_chinese_font()

    # 参数设置
    config = {
        'U': 1.0,  # 来流速度 (m/s)
        'a': 1.0,  # 圆柱半径 (m)
        'rho': 1.2  # 空气密度 (kg/m³)
    }
    Gamma = 2.0  # 环量值 (m²/s)

    # 初始化验证器
    validator = EnhancedLiftValidator(**config)

    # 升力验证
    L_theory = validator.theoretical_lift(Gamma)
    L_num, error = validator.numerical_integration(Gamma, tolerance=0.1)

    print("=== 升力验证结果 ===")
    print(f"理论值: {L_theory:.4f} N/m")
    print(f"数值解: {L_num:.4f} N/m")
    print(f"相对误差: {error:.2f}%")

    # 流场可视化
    validator.flow_visualization(Gamma)
    plt.show()