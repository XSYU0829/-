import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sp
from sympy import sin, Eq, solve, pi, Symbol

# ======================
# 字体配置（解决负号显示问题）
# ======================
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ======================
# 第一部分：公式推导（保持不变）
# ======================
def analytical_solution():
    theta_s = Symbol('θ_s')
    Gamma = Symbol('Γ')
    U = Symbol('U')
    a = Symbol('a')

    equation = Eq(sin(theta_s), -Gamma / (4 * pi * U * a))

    print("建立驻点条件方程并推导解析解：")
    sp.pretty_print(equation)
    print("条件：|Γ| ≤ 4πUa")

    solutions = solve(equation, theta_s)
    print("\n解析解：")
    for i, sol in enumerate(solutions, 1):
        print(f"解 {i}:")
        sp.pretty_print(sol)
    return solutions


# ======================
# 第二部分：可视化（增加字体安全设置）
# ======================
def visualize_results():
    # 参数设置
    U = 1.0
    a = 1.0
    Gamma_crit = 10.0

    # ======================
    # 1. 驻点分岔图（增加字体回退机制）
    # ======================
    plt.figure(figsize=(10, 6))
    Gamma = np.linspace(-10, 10, 1000)

    # 计算角度时强制转换为float类型
    theta_upper = (180 - np.degrees(np.arcsin(-Gamma.astype(float) / (4 * np.pi * U * a)))) % 360
    theta_lower = np.degrees(np.arcsin(-Gamma.astype(float) / (4 * np.pi * U * a))) % 360

    # 使用ASCII负号显示
    plt.plot(Gamma, theta_upper, 'b--', linewidth=1.5, label='上驻点')
    plt.plot(Gamma, theta_lower, 'g-', linewidth=1.5, label='下驻点')

    # 安全显示数学符号
    plt.xlabel('环量Γ (m²/s)', fontname='Microsoft YaHei')
    plt.ylabel('角度位置 (°)', fontname='Microsoft YaHei')

    # 后续绘图代码保持不变...
    plt.xlim(-10, 10)
    plt.ylim(0, 350)
    plt.xticks(np.arange(-10, 11, 5))
    plt.yticks(np.arange(0, 351, 50))
    plt.title('驻点分岔现象', fontname='Microsoft YaHei')
    plt.legend(loc='upper right')
    plt.grid(linestyle='--', alpha=0.5)

    # 安全调用tight_layout
    with np.errstate(invalid='ignore'):
        plt.tight_layout()
    plt.show()

    # ======================
    # 2. 流场可视化函数（增加数值安全处理）
    # ======================
    def plot_streamlines(Gamma, case_type):
        x = np.linspace(-3, 3, 400)
        y = np.linspace(-3, 3, 400)
        X, Y = np.meshgrid(x, y)

        # 数值安全计算
        with np.errstate(divide='ignore', invalid='ignore'):
            r_sq = X ** 2 + Y ** 2
            psi = U * Y * (1 - a ** 2 / r_sq) + (Gamma / (4 * np.pi)) * np.log(r_sq)

        plt.figure(figsize=(8, 8))
        plt.contour(X, Y, np.nan_to_num(psi),  # 处理NaN值
                    levels=np.linspace(-5, 5, 80),
                    colors='black',
                    linewidths=0.5)

        # 绘制安全圆
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=4)

        # 安全设置标题
        plt.title(f'Γ = {float(Gamma):.2f}（{case_type}）',
                  fontname='Microsoft YaHei',
                  fontsize=14)

        # 安全布局
        with np.errstate(invalid='ignore'):
            plt.tight_layout()
        plt.show()

    # ======================
    # 3. 生成工况（确保数值类型安全）
    # ======================
    plot_streamlines(float(6.28), '亚临界')
    plot_streamlines(float(10.0), '临界')
    plot_streamlines(float(12.57), '超临界')


# ======================
# 执行程序
# ======================
if __name__ == "__main__":
    print("=== 公式推导 ===")
    analytical_solution()

    print("\n=== 可视化结果 ===")
    visualize_results()