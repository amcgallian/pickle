# CO2 System Solver
# Andrew McGallian

import matplotlib.pyplot as plt

K1 = 10**(-6.5)             # first dissociation constant (mol/L)
K2 = 10**(-10.8)            # second dissociation constant (mol/L)
K_henry = 10**(1.5) / 1000  # mol/L/atm (Henry's law solubility)
K_water = 10**(-14)         # water dissociation constant (mol/L)^2


def _alpha0(h):
    return h**2 / (h**2 + K1*h + K1*K2)

def _alpha1(h):
    return K1*h / (h**2 + K1*h + K1*K2)

def _alpha2(h):
    return K1*K2 / (h**2 + K1*h + K1*K2)

def _alk_from_dic(DIC_L, h):
    return DIC_L * (_alpha1(h) + 2*_alpha2(h)) + K_water/h - h


def solve_co2_system(alk_m3, DIC_m3, iterations=100):
    """Solve CO2 system given alk and DIC in mol/m³.
    Returns (CO2_m3, CO3_m3, pCO2_ppm, pH).
    """
    alk_L = alk_m3 / 1000
    DIC_L = DIC_m3 / 1000

    pH_lo, pH_hi = 1.0, 14.0
    for _ in range(iterations):
        pH_mid = (pH_lo + pH_hi) / 2
        h = 10**(-pH_mid)
        if _alk_from_dic(DIC_L, h) < alk_L:
            pH_lo = pH_mid
        else:
            pH_hi = pH_mid

    pH = (pH_lo + pH_hi) / 2
    h = 10**(-pH)
    CO2_L = DIC_L * _alpha0(h)
    CO3_L = DIC_L * _alpha2(h)
    pCO2_ppm = (CO2_L / K_henry) * 1e6

    return CO2_L * 1000, CO3_L * 1000, pCO2_ppm, pH


def co2_from_ppm(pCO2_ppm):
    """Return dissolved CO2 concentration (mol/m³) for a given pCO2 (ppm)."""
    CO2_L = K_henry * (pCO2_ppm / 1e6)
    return CO2_L * 1000  # mol/m³


if __name__ == "__main__":
    CO2, CO3, pCO2, pH = solve_co2_system(2.0, 2.0)
    print(f"pH: {pH:.2f}, pCO2: {pCO2:.1f} ppm")
    print(f"Sanity check — expected pH ~8.59, pCO2 ~507 ppm")

    # CO2 drawdown titration
    DIC_vals, pCO2_vals = [], []
    for i in range(50):
        dic = 2.0 - i * 0.02
        _, _, pco2, _ = solve_co2_system(2.0, dic)
        DIC_vals.append(dic)
        pCO2_vals.append(pco2)

    plt.plot(DIC_vals, pCO2_vals)
    plt.xlabel("Total CO2 (mol/m³)")
    plt.ylabel("pCO2 (ppm)")
    plt.title("CO2 Drawdown Titration")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()
