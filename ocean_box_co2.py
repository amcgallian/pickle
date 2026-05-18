# Ocean Box Transient CO2 Uptake Model
# Andrew McGallian
# Simulates a closed air-ocean column responding to an atmospheric CO2 spike,
# with optional CaCO3 compensation.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co2_sys_solver import solve_co2_system, co2_from_ppm
import matplotlib.pyplot as plt

H_ATM   = 8000   # m — atmosphere column height
H_OCEAN = 4000   # m — ocean column depth
K_GAS   = 3 * 365  # m/yr (3 m/day)
K_CACO3 = 10     # m/yr — CaCO3 dissolution rate constant


def ocean_box_model(co2_spike_ppm, caco3_enabled, dt=0.5, duration=1000):
    """
    Simulate transient CO2 uptake in a closed air-ocean box.

    Args:
        co2_spike_ppm:  atmospheric CO2 spike added at t=0 (ppm)
        caco3_enabled:  whether CaCO3 dissolution compensates CO3²⁻ drawdown
        dt:             time step (yr)
        duration:       run length (yr)

    Returns:
        list of lists: [time_points, alk_conc, DIC_conc, CO2_conc, CO3_conc,
                        pCO2_ocean, alk_inv, DIC_inv,
                        atm_pCO2, atm_C_inv, atm_CO2_eq,
                        co2_flux, caco3_flux]
    """
    # Moles of air per m² column (STP: 1 mol = 22.4 L = 0.0224 m³)
    moles_air = H_ATM / 0.0224      # mol/m²
    ppm_per_mol = 1e6 / moles_air   # ppm per (mol/m²)

    # Initial ocean conditions
    alk0, DIC0 = 2.0, 2.0
    CO2_0, CO3_0, pCO2_0, _ = solve_co2_system(alk0, DIC0)

    # Spiked atmosphere
    atm_pCO2 = pCO2_0 + co2_spike_ppm
    atm_C_inv = atm_pCO2 / ppm_per_mol   # mol/m²

    # Ocean inventories (mol/m²)
    alk_inv = alk0 * H_OCEAN
    DIC_inv = DIC0 * H_OCEAN

    # Output lists — seeded with initial values
    time_points   = [0.0]
    alk_conc      = [alk0]
    DIC_conc      = [DIC0]
    CO2_conc      = [CO2_0]
    CO3_conc      = [CO3_0]
    pCO2_ocean    = [pCO2_0]
    alk_inv_list  = [alk_inv]
    DIC_inv_list  = [DIC_inv]
    atm_pCO2_list = [atm_pCO2]
    atm_C_inv_list = [atm_C_inv]
    atm_CO2_eq_list = [co2_from_ppm(atm_pCO2)]
    co2_flux_list  = [0.0]
    caco3_flux_list = [0.0]

    t = 0.0
    for _ in range(int(duration / dt)):
        t += dt
        time_points.append(t)

        # Recalculate ocean carbon speciation
        alk_c = alk_inv / H_OCEAN
        DIC_c = DIC_inv / H_OCEAN
        CO2_c, CO3_c, pCO2_c, _ = solve_co2_system(alk_c, DIC_c)
        pCO2_ocean.append(pCO2_c)
        CO3_conc.append(CO3_c)
        CO2_conc.append(CO2_c)

        # Atmospheric equilibrium CO2 the ocean would like to see
        atm_CO2_eq = co2_from_ppm(atm_pCO2)
        atm_CO2_eq_list.append(atm_CO2_eq)

        # Gas exchange (mol/m²/yr), positive = into ocean
        co2_flux = K_GAS * (atm_CO2_eq - CO2_c)

        # Update atmosphere (closed system)
        atm_C_inv -= co2_flux * dt
        atm_pCO2 = atm_C_inv * ppm_per_mol
        atm_pCO2_list.append(atm_pCO2)
        atm_C_inv_list.append(atm_C_inv)

        # CaCO3 compensation
        if caco3_enabled:
            caco3_flux = K_CACO3 * (CO3_0 - CO3_c)
        else:
            caco3_flux = 0.0

        # Update ocean inventories
        DIC_inv += (co2_flux + caco3_flux) * dt
        alk_inv += 2 * caco3_flux * dt

        alk_conc.append(alk_inv / H_OCEAN)
        DIC_conc.append(DIC_inv / H_OCEAN)
        alk_inv_list.append(alk_inv)
        DIC_inv_list.append(DIC_inv)
        co2_flux_list.append(co2_flux)
        caco3_flux_list.append(caco3_flux)

    return [time_points, alk_conc, DIC_conc, CO2_conc, CO3_conc,
            pCO2_ocean, alk_inv_list, DIC_inv_list,
            atm_pCO2_list, atm_C_inv_list, atm_CO2_eq_list,
            co2_flux_list, caco3_flux_list]


if __name__ == "__main__":
    no_caco3 = ocean_box_model(co2_spike_ppm=200, caco3_enabled=False)
    with_caco3 = ocean_box_model(co2_spike_ppm=200, caco3_enabled=True)

    t_n, _, _, _, CO3_n, pCO2_oc_n, _, _, atm_n, _, atm_eq_n, flux_n, _ = no_caco3
    t_c, alk_c, DIC_c, _, CO3_c, pCO2_oc_c, _, _, atm_c, _, atm_eq_c, flux_c, caco3_c = with_caco3

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # pCO2
    axes[0, 0].plot(t_n, atm_n,      label="Atm (no CaCO3)")
    axes[0, 0].plot(t_n, pCO2_oc_n,  label="Ocean eq (no CaCO3)", linestyle="--")
    axes[0, 0].plot(t_c, atm_c,      label="Atm (with CaCO3)")
    axes[0, 0].plot(t_c, pCO2_oc_c,  label="Ocean eq (with CaCO3)", linestyle="--")
    axes[0, 0].set(xlabel="Time (yr)", ylabel="pCO2 (ppm)", title="Atmospheric & Ocean pCO2")
    axes[0, 0].legend(fontsize=7)

    # CO3
    axes[0, 1].plot(t_n, CO3_n, label="No CaCO3")
    axes[0, 1].plot(t_c, CO3_c, label="With CaCO3")
    axes[0, 1].set(xlabel="Time (yr)", ylabel="CO3²⁻ (mol/m³)", title="Ocean CO3²⁻")
    axes[0, 1].legend()

    # Alk & DIC
    _, alk_n, DIC_n = no_caco3[1], no_caco3[1], no_caco3[2]
    axes[1, 0].plot(t_n, no_caco3[1], label="Alk (no CaCO3)")
    axes[1, 0].plot(t_n, no_caco3[2], label="DIC (no CaCO3)", linestyle="--")
    axes[1, 0].plot(t_c, alk_c,       label="Alk (with CaCO3)")
    axes[1, 0].plot(t_c, DIC_c,       label="DIC (with CaCO3)", linestyle="--")
    axes[1, 0].set(xlabel="Time (yr)", ylabel="mol/m³", title="Ocean Alkalinity & DIC")
    axes[1, 0].legend(fontsize=7)

    # Fluxes
    axes[1, 1].plot(t_n, flux_n,   label="CO2 flux (no CaCO3)")
    axes[1, 1].plot(t_c, flux_c,   label="CO2 flux (with CaCO3)")
    axes[1, 1].plot(t_c, caco3_c,  label="CaCO3 flux", linestyle="--")
    axes[1, 1].set(xlabel="Time (yr)", ylabel="mol/m²/yr", title="CO2 & CaCO3 Fluxes")
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    plt.show()
