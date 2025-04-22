import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import warnings
import matplotlib

# ============================================
# 字体设置修正
# ============================================
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================
# 修正后的驻点迁移分析模块
# ============================================
def stagnation_point_analysis(U=1.0, a=1.0, Gamma_max=15, num_points=200):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    critical_Gamma = 4 * np.pi * U * a
    Gamma_range = np.linspace(0, Gamma_max, num_points)

    # 修正方程符号
    def stagnation_eq(theta, Gamma):
        return np.sin(theta) - Gamma / (4 * np.pi * U * a)

    solutions = []
    for Gamma in Gamma_range:
        if Gamma > critical_Gamma:
            # 超过临界值时寻找表面下方驻点
            theta_surface = 3 * np.pi / 2
            solutions.append((Gamma, theta_surface))
            continue

        # 计算两个可能的解
        C = Gamma / (4 * np.pi * U * a)
        theta1 = np.arcsin(C)
        theta2 = np.pi - theta1

        # 寻找两个解
        for guess in [theta1, theta2]:
            result = root(stagnation_eq, guess, args=(Gamma),
                          method='hybr', tol=1e-8)
            if result.success and abs(result.fun[0]) < 1e-6:
                solutions.append((Gamma, result.x[0]))

    valid_Gammas = [g for g, _ in solutions]
    thetas = [t for _, t in solutions]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(valid_Gammas, np.degrees(thetas), 'b-', lw=2, label='上驻点')
    ax.plot(valid_Gammas[-len(Gamma_range[Gamma_range > critical_Gamma]):],
            [270]*len(Gamma_range[Gamma_range > critical_Gamma]), 'g--', lw=2, label='下表面驻点')
    ax.axvline(critical_Gamma, color='r', ls='--', lw=2,
               label=fr'临界值 $\Gamma_c={critical_Gamma:.2f}$')

    ax.set_xlabel(r'环量 $\Gamma\ (\mathrm{m^2/s})$')
    ax.set_ylabel(r'驻点角度 $\theta\ (\degree)$')
    ax.set_title('驻点迁移特性分析')
    ax.legend()
    plt.show()

    return critical_Gamma

# ============================================
# 压力分布模块（无需修改）
# ============================================
def pressure_distribution(a=1.0, U=1.0, Gamma=5.0):
    theta = np.linspace(0, 2 * np.pi, 500)
    term = Gamma / (2 * np.pi * U * a)
    Cp = 1 - 4 * (np.sin(theta)) ** 2 + term ** 2 - 2 * term * np.sin(theta)

    plt.figure(figsize=(10, 6))
    title = fr'压力分布特性 ($U={U}\, \mathrm{{m/s}},\ a={a}\, \mathrm{{m}}$)'
    plt.title(title, fontsize=12)
    plt.plot(np.degrees(theta), Cp, color='#FF6F00', lw=2)
    plt.xlabel(r'方位角 $\theta\ (\degree)$')
    plt.ylabel(r'压力系数 $C_p$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 新增对称性对比函数
def pressure_symmetry_analysis(a=1.0, U=1.0, critical_Gamma=12.57):
    plt.figure(figsize=(10, 6))
    Gamma_values = [0, 0.5 * critical_Gamma, critical_Gamma]
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    for Gamma, color in zip(Gamma_values, colors):
        theta = np.linspace(0, 2 * np.pi, 500)
        term = Gamma / (2 * np.pi * U * a)
        Cp = 1 - 4 * (np.sin(theta)) ** 2 + term ** 2 - 2 * term * np.sin(theta)
        plt.plot(np.degrees(theta), Cp, lw=2, color=color,
                 label=fr"$\Gamma={Gamma:.1f}$" + (" (临界)" if Gamma == critical_Gamma else ""))

    # 添加对称性标注
    plt.annotate('对称破坏区', (90, -2.5), rotation=90, color='red',
                 ha='center', va='bottom', fontsize=10)
    plt.axvspan(70, 110, alpha=0.2, color='red')

    plt.title("环量对压力对称性的影响", fontsize=12)
    plt.xlabel(r"方位角 $\theta\ (\degree)$")
    plt.ylabel(r"压力系数 $C_p$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    print("执行最终修正版本...")
    critical = stagnation_point_analysis(Gamma_max=15)

    print("\n生成压力分布：")
    pressure_distribution(Gamma=0)
    pressure_distribution(Gamma=critical)