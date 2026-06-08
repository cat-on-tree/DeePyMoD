import numpy as np
from scipy.integrate import odeint

from src.configs.pkpd_registry import tv_pk


def _bateman_concentration(times, dose, bioavailability, volume, ka, ke):
    t = np.asarray(times, dtype=float)
    if ka <= 0:
        return (bioavailability * dose / volume) * np.exp(-ke * t)
    if abs(ka - ke) < 1e-8:
        ka = ke + 1e-3
    pre = (bioavailability * dose * ka) / (volume * (ka - ke))
    c = pre * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(c, 0.0, None)


def simulate_pk_profile(times, route="oral", compartments=1, pk_params=None):
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.array([], dtype=float)
    pk = dict(tv_pk)
    if pk_params:
        pk.update(pk_params)

    route = str(route).lower().strip()
    if route not in {"oral", "bolus"}:
        raise ValueError(f"Unsupported route={route}; expected oral/bolus")
    n_comp = int(compartments)
    if n_comp not in {1, 2, 3}:
        raise ValueError(f"Unsupported compartments={n_comp}; expected 1/2/3")

    t_sorted = np.sort(np.unique(t))
    dose = float(pk.get("D", 100.0))
    bioavailability = float(pk.get("F", 1.0))
    ka = float(pk.get("ka", 1.0 if route == "oral" else 0.0))
    cl = float(pk.get("CL", float(pk.get("ke", 0.2)) * float(pk.get("V", 10.0))))
    v1 = float(pk.get("V", 10.0))
    q2 = float(pk.get("Q2", 0.5))
    q3 = float(pk.get("Q3", 0.3))
    v2 = float(pk.get("V2", 20.0))
    v3 = float(pk.get("V3", 30.0))

    if n_comp == 1:
        ke = cl / max(v1, 1e-8)
        ka_use = ka if route == "oral" else 0.0
        return _bateman_concentration(t, dose, bioavailability, v1, ka_use, ke)

    if route == "oral":
        depot0 = bioavailability * dose
        cp0 = 0.0
    else:
        depot0 = 0.0
        cp0 = bioavailability * dose / max(v1, 1e-8)

    if n_comp == 2:
        z0 = [depot0, cp0, 0.0] if route == "oral" else [cp0, 0.0]

        def rhs_two_oral(z, _tt):
            dep, cp, ct = z
            ddep = -ka * dep
            dcp = (ka / max(v1, 1e-8)) * dep - (cl / max(v1, 1e-8)) * cp - (q2 / max(v1, 1e-8)) * (cp - ct)
            dct = (q2 / max(v2, 1e-8)) * (cp - ct)
            return [ddep, dcp, dct]

        def rhs_two_bolus(z, _tt):
            cp, ct = z
            dcp = -(cl / max(v1, 1e-8)) * cp - (q2 / max(v1, 1e-8)) * (cp - ct)
            dct = (q2 / max(v2, 1e-8)) * (cp - ct)
            return [dcp, dct]

        if route == "oral":
            z = odeint(lambda zz, tt: rhs_two_oral(zz, tt), z0, t_sorted)
            cp = z[:, 1]
        else:
            z = odeint(lambda zz, tt: rhs_two_bolus(zz, tt), z0, t_sorted)
            cp = z[:, 0]
        return np.interp(t, t_sorted, np.clip(cp, 0.0, None))

    z0 = [depot0, cp0, 0.0, 0.0] if route == "oral" else [cp0, 0.0, 0.0]

    def rhs_three_oral(z, _tt):
        dep, cp, ct2, ct3 = z
        ddep = -ka * dep
        dcp = (ka / max(v1, 1e-8)) * dep - (cl / max(v1, 1e-8)) * cp
        dcp -= (q2 / max(v1, 1e-8)) * (cp - ct2) + (q3 / max(v1, 1e-8)) * (cp - ct3)
        dct2 = (q2 / max(v2, 1e-8)) * (cp - ct2)
        dct3 = (q3 / max(v3, 1e-8)) * (cp - ct3)
        return [ddep, dcp, dct2, dct3]

    def rhs_three_bolus(z, _tt):
        cp, ct2, ct3 = z
        dcp = -(cl / max(v1, 1e-8)) * cp
        dcp -= (q2 / max(v1, 1e-8)) * (cp - ct2) + (q3 / max(v1, 1e-8)) * (cp - ct3)
        dct2 = (q2 / max(v2, 1e-8)) * (cp - ct2)
        dct3 = (q3 / max(v3, 1e-8)) * (cp - ct3)
        return [dcp, dct2, dct3]

    if route == "oral":
        z = odeint(lambda zz, tt: rhs_three_oral(zz, tt), z0, t_sorted)
        cp = z[:, 1]
    else:
        z = odeint(lambda zz, tt: rhs_three_bolus(zz, tt), z0, t_sorted)
        cp = z[:, 0]
    return np.interp(t, t_sorted, np.clip(cp, 0.0, None))
