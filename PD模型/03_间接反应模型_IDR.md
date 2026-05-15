# 模型3：间接反应模型（Indirect Response Model, IDR）

简要介绍：
该模型不让药物直接改变药效读出，而是通过影响 `PD1` 的生成端或消除端来起效。这里给出“抑制生成端（Kin）”的经典写法。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = Kin_PD1m*(1 - I(CpR)) - Kout_PD1*PD1
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
I(CpR) = Imax*CpR^gamma1/(IC50^gamma1 + CpR^gamma1)
