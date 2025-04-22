import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tkinter import Tk, font
from tkinter.ttk import Style


# 高级字体配置方案
def configure_scientific_font():
    """多维度字体配置，支持科学符号"""
    # 清除字体缓存（跨平台方案）
    try:
        cache_dir = fm.get_cachedir()
        fm._rebuild()  # 针对matplotlib 3.5+版本的特殊处理
    except:
        pass

    # 创建临时Tk窗口检测字体
    root = Tk()
    root.withdraw()

    # 科学符号兼容字体优先级
    sci_fonts = [
        'Microsoft YaHei',  # 微软雅黑（Win首选）
        'Noto Sans CJK SC',  # 思源黑体（Linux/Mac）
        'SimSun',  # 宋体（备用）
        'Arial Unicode MS'  # 国际符号支持
    ]

    # 动态选择可用字体
    selected_font = next((f for f in sci_fonts
                          if any(f.lower() in x.lower()
                                 for x in font.families())), None)

    # 配置全局字体参数
    if selected_font:
        plt.rcParams.update({
            'font.sans-serif': [selected_font],
            'axes.unicode_minus': False,
            'pdf.fonttype': 42  # 确保PDF输出兼容
        })
        # 同步配置Tkinter字体
        style = Style()
        style.configure('.', font=(selected_font, 12))
    else:
        raise EnvironmentError("请安装Microsoft YaHei或思源黑体字体包")

    root.destroy()


# 执行高级字体配置
configure_scientific_font()

# ========== 流体力学计算核心 ==========
U = 1.0  # 来流速度 [m/s]
a = 1.0  # 圆柱半径 [m]
n = 1.8  # 安全系数

Γ_critical = 4 * np.pi * U * a  # 临界环量
Γ_safe = Γ_critical / n  # 工程安全环量

# 参数扫描范围
Γ_values = np.linspace(0, 6 * np.pi * U * a, 300)

# 误差模型（含科学符号计算）
error_coeff = 20 / (6 * np.pi - 4 * np.pi) ** 2
error = np.where(Γ_values > Γ_critical,
                 error_coeff * (Γ_values - Γ_critical) ** 2, 0)

# ========== 工程级可视化 ==========
plt.figure(figsize=(10, 6), dpi=120)
ax = plt.gca()

# 主曲线绘制
main_plot, = ax.plot(Γ_values, error, 'r-',
                     label=r'势流理论误差 ($\Delta \Gamma^2$)')

# 关键线标注
ax.axvline(Γ_critical, color='b', ls='--',
           label=r'临界环量 $4\pi Ua$')
ax.axvline(Γ_safe, color='g', ls='-.',
           label=f'安全环量 (n={n})')

# 误差区域填充
ax.fill_between(Γ_values, error, 0, where=(Γ_values > Γ_critical),
                color='#FFAAAA', alpha=0.3)

# 专业标注系统
ax.annotate(r'$\Gamma_{\mathrm{safe}} = \frac{4\pi U a}{n}$',
            xy=(Γ_safe, 19), xytext=(Γ_safe + 0.5, 15),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            fontsize=12, backgroundcolor='#FFFFFFAA')

ax.set_title(r'任务3.1 安全环量范围确定 ($m^2/s$)', pad=20)
ax.set_xlabel(r'环量 $\Gamma$ [m²/s]', labelpad=10)
ax.set_ylabel('理论误差 [%]', labelpad=10)

# 科学坐标轴配置
ax.set_xticks([0, Γ_safe, Γ_critical, 6 * np.pi])
ax.set_xticklabels(['0', r'$\Gamma_{\mathrm{safe}}$',
                    r'$4\pi Ua$', r'$6\pi Ua$'],
                   fontsize=10)
ax.set_xlim(0, 6 * np.pi)
ax.set_ylim(0, 20)

# 网格及图例优化
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend(loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig('output.pdf', bbox_inches='tight')  # 支持矢量输出
plt.show()