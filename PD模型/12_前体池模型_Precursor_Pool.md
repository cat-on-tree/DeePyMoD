# 模型12：前体池模型（Precursor Pool Model）

简要介绍：
该模型在主药效状态之前引入一个前体池状态 `P`，用于描述药物先作用于前体生成或耗竭，再由前体池变化传递到最终药效指标的场景。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dP/dt    = Kin_P*(1 + E1(CpR)) - ktr*P
(6)  dPD1/dt  = ktr*P - Kout_PD1*PD1
(7)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E1(CpR) = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
