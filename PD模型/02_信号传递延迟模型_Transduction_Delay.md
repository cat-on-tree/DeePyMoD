# 模型2：信号传递延迟模型（Transduction Delay Model）

简要介绍：
该模型在上游药物作用输入与主药效之间加入 3 个转导室（`T1`、`T2`、`T3`），用于描述药效迟滞和相位延后。这里以上游驱动量 `E_input(t)` 进入延迟链，延迟末端 `T3` 驱动 `PD1`。

方程组：
(1)  dA1/dt   = -ka*A1
(2)  dCp/dt   = (ka/V2)*A1 - keD*Cp - Kon*Cp*R + Koff*CpR
(3)  dR/dt    = Kin_R(t) - Kout_R*R - Kon*Cp*R + Koff*CpR
(4)  dCpR/dt  = Kon*Cp*R - Koff*CpR - keDR*CpR
(5)  dT1/dt   = kt1*(E_input(t) - T1)
(6)  dT2/dt   = kt1*(T1 - T2)
(7)  dT3/dt   = kt1*(T2 - T3)
(8)  dPD1/dt  = Kin_PD1m*(1 + E1(T3)) - Kout_PD1*PD1
(9)  dPD2/dt  = Kout_PD1*PD1 - Kout_PD2*PD2

其中：
E1(T3) = Emax1*T3^gamma1/(EC50^gamma1 + T3^gamma1)
