# OAE Mixing Model
# Andrew McGallian
# Simulates a surface ocean mixed layer with gas exchange, deepwater mixing,
# and addition of alkalinity/DIC (OAE scenarios).

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co2_sys_solver import solve_co2_system, co2_from_ppm
import matplotlib.pyplot as plt

MIXED_LAYER_DEPTH = 100   # m
K_GAS = 3 * 365           # m/yr (3 m/day gas exchange piston velocity)


def oae_mixing_model(alk_add_rate, DIC_add_rate, deepwater_exchange,
                     dt=0.1, duration=20):
    """
    Simulate OAE in a surface ocean mixed layer.

    Args:
        alk_add_rate:       alkalinity addition rate (mol/m²/yr)
        DIC_add_rate:       total CO2 addition rate (mol/m²/yr)
        deepwater_exchange: exchange constant with deep water (m/yr)
        dt:                 time step (yr)
        duration:           run length (yr)

    Returns:
        list of lists: [time_points, DIC_conc, alk_conc, pCO2_ocean, co2_flux]
    """
    h = MIXED_LAYER_DEPTH

    # Initial conditions
    alk0, DIC0 = 2.0, 2.0  # mol/m³
    CO2_0, _, pCO2_0, _ = solve_co2_system(alk0, DIC0)
    CO2_atm_eq = CO2_0  # atmospheric equilibrium CO2 stays fixed

    # Inventories (mol/m²)
    alk_inv = alk0 * h
    DIC_inv = DIC0 * h

    # Output lists
    time_points  = [0.0]
    DIC_conc     = [DIC0]
    alk_conc     = [alk0]
    pCO2_ocean   = [pCO2_0]
    co2_flux_list = [0.0]

    t = 0.0
    for _ in range(int(duration / dt)):
        t += dt
        time_points.append(t)

        # Current ocean chemistry
        alk_c = alk_inv / h
        DIC_c = DIC_inv / h
        CO2_c, _, pCO2_c, _ = solve_co2_system(alk_c, DIC_c)
        pCO2_ocean.append(pCO2_c)

        # Gas exchange flux (mol/m²/yr), positive = into ocean
        co2_flux = K_GAS * (CO2_atm_eq - CO2_c)
        DIC_inv += co2_flux * dt

        # Deepwater exchange (restores toward initial values)
        DIC_exchange = deepwater_exchange * (DIC0 - DIC_c)
        alk_exchange = deepwater_exchange * (alk0 - alk_c)
        DIC_inv += DIC_exchange * dt
        alk_inv += alk_exchange * dt

        # Chemistry addition (OAE / fertilization)
        DIC_inv += DIC_add_rate * dt
        alk_inv += alk_add_rate * dt

        DIC_conc.append(DIC_inv / h)
        alk_conc.append(alk_inv / h)
        co2_flux_list.append(co2_flux)

    return [time_points, DIC_conc, alk_conc, pCO2_ocean, co2_flux_list]


if __name__ == "__main__":
    # Scenario 1: CaO addition, no deepwater exchange
    s1 = oae_mixing_model(alk_add_rate=0.1, DIC_add_rate=0.0,  deepwater_exchange=0)
    # Scenario 2: CaCO3 addition (alk:DIC = 2:1), no mixing
    s2 = oae_mixing_model(alk_add_rate=0.1, DIC_add_rate=0.05, deepwater_exchange=0)
    # Scenario 3: CaO + deepwater exchange
    s3 = oae_mixing_model(alk_add_rate=0.1, DIC_add_rate=0.0,  deepwater_exchange=100)
    # Scenario 4: Biological pump (negative DIC flux), mixing on
    s4 = oae_mixing_model(alk_add_rate=0.0, DIC_add_rate=-0.1, deepwater_exchange=100)
    # Scenario 5: Bicarbonate addition (equal alk and DIC), mixing on
    s5 = oae_mixing_model(alk_add_rate=0.1, DIC_add_rate=0.1,  deepwater_exchange=100)

    labels = [
        "CaO, no mixing",
        "CaCO3, no mixing",
        "CaO + mixing",
        "Bio pump + mixing",
        "Bicarbonate + mixing",
    ]
    scenarios = [s1, s2, s3, s4, s5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_dic, ax_alk, ax_pco2, ax_flux = axes.flatten()

    for s, label in zip(scenarios, labels):
        t, dic, alk, pco2, flux = s
        ax_dic.plot(t, dic, label=label)
        ax_alk.plot(t, alk, label=label)
        ax_pco2.plot(t, pco2, label=label)
        ax_flux.plot(t, flux, label=label)

    ax_dic.set(xlabel="Time (yr)", ylabel="DIC (mol/m³)", title="Total CO2 Concentration")
    ax_alk.set(xlabel="Time (yr)", ylabel="Alkalinity (mol/m³)", title="Alkalinity Concentration")
    ax_pco2.set(xlabel="Time (yr)", ylabel="pCO2 (ppm)", title="Ocean Equilibrium pCO2")
    ax_flux.set(xlabel="Time (yr)", ylabel="CO2 flux (mol/m²/yr)", title="CO2 Invasion Flux")

    for ax in axes.flatten():
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()
