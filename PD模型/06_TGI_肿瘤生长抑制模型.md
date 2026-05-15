# 模型6：肿瘤生长抑制模型（Tumor Growth Inhibition, TGI）

简要介绍：
该模型将 `PD1` 解释为疾病负荷或肿瘤体积，由自然生长项与药物杀伤项共同决定，并可带耐药衰减。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = k_grow*PD1 - k_kill(CpR,t)*PD1
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
k_kill(CpR,t) = E_kill(CpR)*exp(-lambda*t)
E_kill(CpR)   = Emax_kill*CpR^gamma_kill/(EC50_kill^gamma_kill + CpR^gamma_kill)
