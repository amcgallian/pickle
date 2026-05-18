# Mineral Dissolution Kinetics Model
# Andrew McGallian
# Simulates pH-dependent dissolution of CaO and CaCO3 in soil water
# starting from equilibrium with high-CO2 soil gas.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co2_sys_solver import solve_co2_system, co2_from_ppm
import matplotlib.pyplot as plt


def mineral_dissolution_model(soil_pCO2_ppm, k_CaO, k_CaCO3, dt=0.1, duration=10):
    """
    Simulate mineral dissolution kinetics in soil water.

    Dissolution rates scale with [H+] relative to pH 5:
        rate = k * 10^(5 - pH)   [mol/m³/yr]

    CaO dissolving:   adds 2 eq alk, no DIC change
    CaCO3 dissolving: adds 2 eq alk, 1 mol DIC

    Args:
        soil_pCO2_ppm: soil gas pCO2 (ppm)
        k_CaO:         CaO rate constant at pH 5 (mol/m³/yr)
        k_CaCO3:       CaCO3 rate constant at pH 5 (mol/m³/yr)
        dt:            time step (yr)
        duration:      run length (yr)

    Returns:
        list of lists: [time_points, DIC_conc, alk_conc, pH_list,
                        rate_CaO_list, rate_CaCO3_list]
    """
    # Initial conditions
    CO2_eq = co2_from_ppm(soil_pCO2_ppm)  # mol/m³, CO2 dissolved from soil gas
    DIC = CO2_eq                            # initially all dissolved CO2
    alk = 0.0                               # no alkalinity in pure rainwater

    _, _, _, pH = solve_co2_system(alk if alk > 0 else 1e-10, DIC)

    # Output lists — seeded with initial values
    time_points    = [0.0]
    DIC_conc       = [DIC]
    alk_conc       = [alk]
    pH_list        = [pH]
    rate_CaO_list  = [0.0]
    rate_CaCO3_list = [0.0]

    t = 0.0
    for _ in range(int(duration / dt)):
        t += dt
        time_points.append(t)

        # Dissolution rates scale with [H+] vs pH 5
        r_CaO   = k_CaO   * 10**(5 - pH)
        r_CaCO3 = k_CaCO3 * 10**(5 - pH)

        # Update concentrations
        alk += (2 * r_CaO + 2 * r_CaCO3) * dt
        DIC += r_CaCO3 * dt

        # Recalculate pH — guard against alk <= 0
        _, _, _, pH = solve_co2_system(max(alk, 1e-10), DIC)

        DIC_conc.append(DIC)
        alk_conc.append(alk)
        pH_list.append(pH)
        rate_CaO_list.append(r_CaO)
        rate_CaCO3_list.append(r_CaCO3)

    return [time_points, DIC_conc, alk_conc, pH_list, rate_CaO_list, rate_CaCO3_list]


if __name__ == "__main__":
    SOIL_PCO2 = 4000   # ppm (~10x atmospheric)
    K_CAO     = 1e-3   # mol/m³/yr at pH 5
    K_CACO3   = 1e-2   # 10x faster

    # Three experiments
    cao_only   = mineral_dissolution_model(SOIL_PCO2, k_CaO=K_CAO,   k_CaCO3=0)
    caco3_only = mineral_dissolution_model(SOIL_PCO2, k_CaO=0,        k_CaCO3=K_CACO3)
    both       = mineral_dissolution_model(SOIL_PCO2, k_CaO=K_CAO,   k_CaCO3=K_CACO3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    run_labels = ["CaO only", "CaCO3 only", "Both"]
    runs = [cao_only, caco3_only, both]

    for run, label in zip(runs, run_labels):
        t, dic, alk, pH, r_cao, r_caco3 = run
        axes[0, 0].plot(t, pH,  label=label)
        axes[0, 1].plot(t, alk, label=label)
        axes[1, 0].plot(t, dic, label=label)
        axes[1, 1].plot(t, r_cao, label=f"{label} — CaO rate")

    axes[0, 0].set(xlabel="Time (yr)", ylabel="pH", title="Solution pH")
    axes[0, 1].set(xlabel="Time (yr)", ylabel="Alkalinity (mol/m³)", title="Alkalinity")
    axes[1, 0].set(xlabel="Time (yr)", ylabel="DIC (mol/m³)", title="Total CO2 (DIC)")
    axes[1, 1].set(xlabel="Time (yr)", ylabel="Rate (mol/m³/yr)", title="CaO Dissolution Rate (= CO2 Uptake Rate)")

    for ax in axes.flatten():
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
