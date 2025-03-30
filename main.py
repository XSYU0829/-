from sympy import symbols, diff, re, im, simplify, exp, I

# 定义符号变量
r, theta = symbols('r theta', real=True)
a, U = symbols('a U', real=True, positive=True)

# 极坐标复数表示
z = r * exp(I * theta)  # 使用 sympy 的 exp 函数和虚数单位 I

# 复势函数
Phi = U * (z + a**2 / z)

# 速度势函数和流函数
phi = re(Phi)
psi = im(Phi)

# 计算偏导数
dphi_dr = diff(phi, r)
dpsi_dtheta = diff(psi, theta)
cauchy_riemann_eq1 = simplify(dphi_dr - (1/r) * dpsi_dtheta)  # 应等于0

dpsi_dr = diff(psi, r)
dphi_dtheta = diff(phi, theta)
cauchy_riemann_eq2 = simplify(dpsi_dr + (1/r) * dphi_dtheta)  # 应等于0

print("柯西-黎曼方程验证结果：")
print("方程1残差：", cauchy_riemann_eq1)  # 输出应为0
print("方程2残差：", cauchy_riemann_eq2)  # 输出应为0