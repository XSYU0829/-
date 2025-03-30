import sympy as sp
import numpy as np


# ==============================================
# 理论验证部分：符号推导柯西-黎曼方程
# ==============================================
def theoretical_verification():
    # 定义符号变量(均为正实数)
    r, θ, U, a = sp.symbols('r θ U a', positive=True)

    # 定义复势函数 Φ(z) = U*(z + a²/z)
    z = r * sp.exp(sp.I * θ)
    phi = U * (z + a ** 2 / z)

    # 分解实部虚部
    u = sp.re(phi).simplify()  # 实部
    v = sp.im(phi).simplify()  # 虚部

    # 极坐标柯西-黎曼方程
    # 方程1: ∂u/∂r = (1/r) * ∂v/∂θ
    # 方程2: ∂v/∂r = -(1/r) * ∂u/∂θ

    # 计算偏导数
    u_r = sp.diff(u, r)
    v_θ = sp.diff(v, θ)
    eq1 = u_r - (1 / r) * v_θ

    v_r = sp.diff(v, r)
    u_θ = sp.diff(u, θ)
    eq2 = v_r + (1 / r) * u_θ

    # 简化验证结果
    eq1_simplified = sp.trigsimp(eq1)
    eq2_simplified = sp.trigsimp(eq2)

    print("理论验证结果：")
    print(f"柯西-黎曼方程1残差: {eq1_simplified}")
    print(f"柯西-黎曼方程2残差: {eq2_simplified}\n")


# ==============================================
# 数值验证部分：计算最大残差
# ==============================================
def numerical_verification():
    # 定义符号变量
    r, θ, U, a = sp.symbols('r θ U a', positive=True)

    # 定义复势及其导数
    z = r * sp.exp(sp.I * θ)
    phi = U * (z + a ** 2 / z)
    u = sp.re(phi)
    v = sp.im(phi)

    # 计算偏导数
    derivatives = {
        'u_r': sp.diff(u, r),
        'u_θ': sp.diff(u, θ),
        'v_r': sp.diff(v, r),
        'v_θ': sp.diff(v, θ)
    }

    # 定义柯西-黎曼方程残差
    cr_residuals = [
        derivatives['u_r'] - (1 / r) * derivatives['v_θ'],
        derivatives['v_r'] + (1 / r) * derivatives['u_θ']
    ]

    # 转换为数值计算函数
    cr_funcs = [sp.lambdify((r, θ, U, a), res, 'numpy')
                for res in cr_residuals]

    # 设置采样参数
    U_val = 1.0  # 流速参数
    a_val = 1.0  # 特征半径
    r_vals = np.linspace(2, 10, 100)  # r > a区域采样
    θ_vals = np.linspace(0, 2 * np.pi, 100)

    # 计算最大残差
    max_res = 0.0
    for r_val in r_vals:
        for θ_val in θ_vals:
            # 计算两个方程的残差
            residuals = [abs(f(r_val, θ_val, U_val, a_val))
                         for f in cr_funcs]
            current_max = max(residuals)
            if current_max > max_res:
                max_res = current_max

    print("数值验证结果：")
    print(f"最大残差: {max_res:.2e}")
    print("验证结论：" + ("通过" if max_res < 1e-6 else "未通过"))


# ==============================================
# 执行验证流程
# ==============================================
if __name__ == "__main__":
    theoretical_verification()
    numerical_verification()