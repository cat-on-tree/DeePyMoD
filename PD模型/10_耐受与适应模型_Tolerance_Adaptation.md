# 模型10：耐受与适应模型（Tolerance / Adaptation Model）

简要介绍：
该模型在药物主效应之外引入一个缓慢变化的适应状态，用于描述持续给药后药效逐渐减弱、停药后可能恢复的耐受或适应过程。

方程组：
(1)  dA1/dt    = -ka*A1
(2)  dCp/dt    = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt     = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt   = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dTol/dt   = k_in_tol*(1 + S_tol(CpR)) - k_out_tol*Tol
(6)  dPD1/dt   = Kin_PD1m*(1 + E1(CpR))/(1 + Tol) - Kout_PD1*PD1
(7)  dPD2/dt   = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E1(CpR)    = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
S_tol(CpR) = Smax_tol*CpR^gamma_tol/(SC50_tol^gamma_tol + CpR^gamma_tol)
