# 模型8：生物相/效应室模型（Biophase / Effect Compartment Model）

简要介绍：
该模型在上游药物作用输入与药效之间增加效应室浓度 `Ce`，用于表达分布延迟加直接效应的场景。

方程组：
(1)  dA1/dt        = -ka*A1
(2)  dCp/dt        = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt         = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt       = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dCe/dt        = ke0*(CpR - Ce)
(6)  PD1           = E0 + Emax1*Ce^gamma1/(EC50^gamma1 + Ce^gamma1)
(7)  dPD2/dt       = Kout_PD1*PD1 - Kout_PD2*PD2
