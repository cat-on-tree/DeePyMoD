# 模型11：药物相互作用模型（Drug Interaction Model）

简要介绍：
该模型用于描述两种药物共同存在时，一种相互作用药物改变主药药效强度的场景。这里给出竞争性减弱主药效应的示例，即相互作用药物使主药的表观敏感性下降。

方程组：
(1)  dA1/dt      = -ka*A1
(2)  dCp/dt      = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt       = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt     = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dAint/dt    = -ka_int*Aint
(6)  dCint/dt    = (ka_int/Vint)*Aint - ke_int*Cint
(7)  dPD1/dt     = Kin_PD1m*(1 + Eint(CpR,Cint)) - Kout_PD1*PD1
(8)  dPD2/dt     = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
Eint(CpR,Cint) = Emax1*CpR^gamma1/((EC50*(1 + Cint/KI))^gamma1 + CpR^gamma1)
