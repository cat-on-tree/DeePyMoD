# 模型13：抗体类 PK/PD 模型（Antibody PK/PD Model）

简要介绍：
该模型用于描述抗体类药物在血浆与组织间分布、与靶点结合并进一步产生药效响应的场景，适合表征蛋白/抗体药物的较慢处置与较长作用过程。

方程组：
(1)  dA1/dt    = -ka*A1
(2)  dCp/dt    = (ka/V2)*A1 - CLp*Cp/V2 - Q*(Cp-Ct)/V2 - Kon*Cp*R + Koff*CpR
(3)  dCt/dt    = Q*(Cp-Ct)/Vt
(4)  dR/dt     = Kin_R - Kout_R*R - Kon*Cp*R + Koff*CpR
(5)  dCpR/dt   = Kon*Cp*R - Koff*CpR - keDR*CpR
(6)  dPD1/dt   = Kin_PD1m*(1 + E1(CpR)) - Kout_PD1*PD1
(7)  dPD2/dt   = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E1(CpR) = Emax1*CpR^gamma1/(EC50^gamma1 + CpR^gamma1)
