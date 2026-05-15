# 模型1：靶点介导的药物处置模型（Target-Mediated Drug Disposition, TMDD）

简要介绍：
该模型强调药物与靶点受体的可逆结合过程，以及结合态复合物对后续药效的驱动作用。这里采用药物-受体复合物 `CpR` 直接驱动 `PD1`，再由 `PD1` 传递到 `PD2`。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = Kin_PD1m*(1 + E1(CpR)) - Kout_PD1*PD1
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E1(CpR) = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
