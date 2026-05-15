# 模型9：疾病进展模型（Disease Progression Model）

简要介绍：
该模型将 `PD1` 解释为疾病严重程度、疾病负荷或疾病进展指标，由疾病自然进展与药物干预共同决定，用于描述一般疾病状态随时间变化的场景。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dPD1/dt  = k_prog - k_rem*PD1 - E_dis(CpR)
(6)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E_dis(CpR) = Emax_dis*CpR^gamma_dis/(EC50_dis^gamma_dis + CpR^gamma_dis)
