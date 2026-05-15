# 模型7：直接效应模型（Direct Effect Model）

简要介绍：
该模型假设药效读出由上游驱动量即时决定，采用直接效应的 `Emax/Hill` 代数表达，不经过主药效周转通路。

方程组：
(1)  dA1/dt        = -ka*A1
(2)  dCp/dt        = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt         = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt       = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  PD1           = E0 + E_direct(CpR)
(6)  dPD2/dt       = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E_direct(CpR) = Emax_direct*CpR^gamma_direct/(EC50_direct^gamma_direct + CpR^gamma_direct)
