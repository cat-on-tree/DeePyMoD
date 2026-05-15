# 模型5：昼夜节律调控模型（Circadian Regulation Model）

简要介绍：
该模型在受体生成端或主药效生成端引入 24 小时节律项，用于描述基础生理节律与药物作用叠加的场景。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = Kin_PD1(t)*(1 + E1(CpR)) - Kout_PD1*PD1
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
Kin_R(t)   = Kin_Rm + Kin_Rb*cos((t - phi1)*2*pi/24)
Kin_PD1(t) = Kin_PD1m + Kin_PD1b*cos((t - phi2)*2*pi/24)
E1(CpR)    = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
