from sympy import symbols, diff, simplify
def verify_cr_equations(phi, psi, x, y):
    """验证复势函数的实虚部是否满足柯西-黎曼方程"""
    # 定义符号变量
    x_sym, y_sym = symbols('x y')

    # 计算偏导数
    dphi_dx = diff(phi, x_sym)
    dphi_dy = diff(phi, y_sym)
    dpsi_dx = diff(psi, x_sym)
    dpsi_dy = diff(psi, y_sym)

    # 验证方程
    cr1 = simplify(dphi_dx - dpsi_dy)
    cr2 = simplify(dphi_dy + dpsi_dx)

    # 返回最大残差
    max_error = max(abs(cr1.subs({x_sym: x, y_sym: y})),
                    abs(cr2.subs({x_sym: x, y_sym: y})))
    return max_error