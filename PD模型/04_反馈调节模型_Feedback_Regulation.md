# 模型4：反馈调节模型（Feedback Regulation Model）

简要介绍：
该模型在上游药物作用到 `PD1`、`PD2` 的主链之外，引入 `PD2` 经延迟链反馈调节 `PD1` 生成端，用于描述耐受、反跳等适应过程。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = Kin_PD1m*(1 + E1(CpR))*(1 + E2(T6)) - Kout_PD1*PD1
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2
(7)  dT4/dt   = kt2*(PD2 - T4)
(8)  dT5/dt   = kt2*(T4 - T5)
(9)  dT6/dt   = kt2*(T5 - T6)

其中：
E1(CpR) = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
E2(T6)  = Emax2*T6^gamma2/(EC50_2^gamma2 + T6^gamma2)
