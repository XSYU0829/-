# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ============================================
# 全局字体配置
# ============================================
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
    'axes.unicode_minus': False,
    'text.usetex': False
})


# ============================================
# 驻点迁移分析函数（新增函数定义）
# ============================================
def stagnation_analysis(U=1.0, a=1.0):
    """驻点位置随环量变化分析"""
    critical_Gamma = 4 * np.pi * U * a
    Gamma_range = np.linspace(-critical_Gamma * 1.1, critical_Gamma * 1.1, 150)

    def stagnation_eq(theta, Gamma):
        term = Gamma / (4 * np.pi * U * a)
        return 1 - 4 * (np.sin(theta)) ** 2 + term ** 2 - 2 * term * np.sin(theta)

    solutions = []
    for Gamma in Gamma_range:
        initial_guesses = np.linspace(-np.pi, np.pi, 12)
        for guess in initial_guesses:
            sol = fsolve(stagnation_eq, guess, args=(Gamma,), xtol=1e-6)
            if abs(stagnation_eq(sol, Gamma)) < 1e-4:
                solutions.append((Gamma, sol[0]))
                break

    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in solutions], np.degrees([x[1] for x in solutions]), 'b-', label='驻点位置')
    plt.axvline(critical_Gamma, color='r', linestyle='--', label=rf'临界环量 $\Gamma_c = {critical_Gamma:.1f}$')
    plt.axvline(-critical_Gamma, color='r', linestyle='--')
    plt.title('驻点迁移特性分析')
    plt.xlabel(r'环量 $\Gamma$ (m²/s)')
    plt.ylabel(r'驻点角度 $\theta$ (°)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================
# 压力场分析函数
# ============================================
def pressure_analysis(a=1.0, U=1.0):
    """圆柱表面压力分布计算"""
    theta = np.linspace(0, 2 * np.pi, 360)

    plt.figure(figsize=(12, 6))
    cases = [
        (-4 * np.pi * U * a, '逆向临界'),
        (0, '无环量'),
        (4 * np.pi * U * a, '正向临界')
    ]

    for Gamma, label in cases:
        term = Gamma / (2 * np.pi * U * a)
        Cp = 1 - 4 * np.sin(theta) ** 2 + term ** 2 - 2 * term * np.sin(theta)
        plt.plot(np.degrees(theta), Cp, label=rf'{label} $\Gamma={Gamma / np.pi:.1f}\pi$')

    plt.title('圆柱表面压力分布')
    plt.xlabel(r'方位角 $\theta$ (°)')
    plt.ylabel('压力系数 Cp')
    plt.xticks(np.arange(0, 361, 45))
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================
# 流场动画类（修正后的版本）
# ============================================
class FlowAnimator:
    def __init__(self, U=1.0, a=1.0):
        self.U = U
        self.a = a
        self.x = np.linspace(-3, 3, 150)
        self.y = np.linspace(-3, 3, 150)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect('equal')

    def stream_function(self, Gamma):
        r = np.sqrt(self.X ** 2 + self.Y ** 2)
        theta = np.arctan2(self.Y, self.X)
        return (self.U * r * np.sin(theta) * (1 - (self.a ** 2) / (r ** 2))
                - (Gamma / (2 * np.pi)) * np.log(np.clip(r / self.a, 1e-6, None)))

    def update(self, Gamma):
        self.ax.clear()
        psi = self.stream_function(Gamma)
        u = np.gradient(psi, self.y, axis=0)
        v = -np.gradient(psi, self.x, axis=1)

        self.ax.streamplot(self.x, self.y, u, v, density=1.5,
                           color='deepskyblue', linewidth=1.2)
        self.ax.add_patch(Circle((0, 0), self.a, color='gray', alpha=0.6))
        self.ax.set_title(rf'流场演化  $\Gamma = {Gamma / np.pi:.1f}\pi$')
        return self.ax

    def animate(self):
        Gamma_range = np.linspace(-4 * np.pi * self.U * self.a,
                                  4 * np.pi * self.U * self.a, 120)
        anim = FuncAnimation(self.fig, self.update, frames=Gamma_range,
                             interval=50, blit=False)
        return anim


# ============================================
# 统一主程序入口（关键修正点）
# ============================================
if __name__ == "__main__":
    print("启动流体力学分析系统...")

    # 阶段1：驻点迁移分析
    print("\n[阶段1] 驻点迁移分析...")
    stagnation_analysis()

    # 阶段2：压力场分析
    print("\n[阶段2] 压力场分析...")
    pressure_analysis()

    # 阶段3：流场动画
    print("\n[阶段3] 生成流场动画...")
    animator = FlowAnimator()
    animation = animator.animate()

plt.show()
