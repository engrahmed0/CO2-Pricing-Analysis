# simulate_system.py

import pandas as pd
import linopy

# Load and prepare data
chunks = pd.read_csv("electricity_demand_clean.csv", parse_dates=["timestamp"], chunksize=5000)
elec_demand = pd.concat(chunks)
elec_demand = elec_demand.rename(columns={"timestamp": "time"})
elec_demand['time'] = elec_demand['time'] + pd.DateOffset(years=4)

pv_df = pd.read_csv("ninja_pv_51.1638_10.4478_corrected.csv", skiprows=3)
pv_df.columns = ["time", "local_time", "electricity", "irr_direct", "irr_diffuse", "temp"]
pv_df["time"] = pd.to_datetime(pv_df["time"])
pv_df.rename(columns={"electricity": "pv_capacity_factor"}, inplace=True)

heat_df = pd.read_csv("ninja_demand_51.1638_10.4478_uncorrected.csv", skiprows=3)
heat_df.columns = ["time", "local_time", "total_demand", "heating_demand", "cooling_demand"]
heat_df["time"] = pd.to_datetime(heat_df["time"])

# Merge
df = elec_demand.merge(pv_df[["time", "pv_capacity_factor"]], on="time", how="inner")
df = df.merge(heat_df[["time", "heating_demand"]], on="time", how="inner")
assert not df.isna().any().any(), "Merged dataframe contains NaNs"

# Constants
grid_price = 0.30
co2_grid = 0.401
co2_gas = 0.202
heat_pump_COP = 3.0
battery_eff = 0.9

pv_capex = 97.653
battery_capex = 50.0
heat_pump_capex = 51.6992

battery_power_limit = 5
heat_pump_max_power = 5

n_hours = len(df)
T = list(range(n_hours))
demand_electricity = df["electricity_demand_kWh"].values
heat_demand = df["heating_demand"].values
pv_cf = df["pv_capacity_factor"].values

scenarios = [0, 50, 100]
all_results = []

for co2_price_eur_ton in scenarios:
    co2_price = co2_price_eur_ton / 1000
    m = linopy.Model()

    # Variables
    grid_import = m.add_variables(name="grid_import", lower=0, coords=[T])
    pv_gen = m.add_variables(name="pv_gen", lower=0, coords=[T])
    battery_charge = m.add_variables(name="battery_charge", lower=0, coords=[T])
    battery_discharge = m.add_variables(name="battery_discharge", lower=0, coords=[T])
    battery_soc = m.add_variables(name="battery_soc", lower=0, coords=[T])
    gas_boiler_heat = m.add_variables(name="gas_boiler_heat", lower=0, coords=[T])
    heat_pump_electricity = m.add_variables(name="heat_pump_el", lower=0, coords=[T])
    battery_capacity = m.add_variables(name="battery_capacity", lower=0)
    pv_capacity = m.add_variables(name="pv_capacity", lower=0)
    heat_pump_capacity = m.add_variables(name="heat_pump_capacity", lower=0)

    m.add_constraints((battery_soc.isel(dim_0=0) - battery_charge.isel(dim_0=0) * battery_eff) == 0)

    for t in range(1, n_hours):
        m.add_constraints(
            (battery_soc.isel(dim_0=t) - battery_soc.loc[t - 1] - battery_charge.isel(dim_0=t) * battery_eff + battery_discharge.isel(dim_0=t) / battery_eff) == 0
        )

    for t in T:
        m.add_constraints(
            grid_import.isel(dim_0=t) + pv_gen.isel(dim_0=t) + battery_discharge.isel(dim_0=t)
            - battery_charge.isel(dim_0=t) - heat_pump_electricity.isel(dim_0=t) == demand_electricity[t]
        )
        m.add_constraints(pv_gen.isel(dim_0=t) <= pv_capacity * pv_cf[t])
        m.add_constraints(battery_soc.isel(dim_0=t) <= battery_capacity)
        m.add_constraints(battery_charge.isel(dim_0=t) <= battery_power_limit)
        m.add_constraints(battery_discharge.isel(dim_0=t) <= battery_power_limit)
        m.add_constraints(gas_boiler_heat.isel(dim_0=t) + heat_pump_electricity.isel(dim_0=t) * heat_pump_COP == heat_demand[t])
        m.add_constraints(heat_pump_electricity.isel(dim_0=t) <= heat_pump_capacity)

    m.objective = (
        (grid_import * grid_price).sum() +
        (grid_import * co2_grid * co2_price).sum() +
        (gas_boiler_heat * co2_gas * co2_price).sum() +
        pv_capacity * pv_capex +
        battery_capacity * battery_capex +
        heat_pump_capacity * heat_pump_capex
    )

    m.solve()
    results = m.solution
    result_df = pd.DataFrame({k: results[k].values for k in results})
    result_df["time"] = df["time"]
    result_df["co2_price"] = co2_price_eur_ton
    result_df["cost_grid"] = result_df["grid_import"] * grid_price
    result_df["co2_emissions"] = result_df["grid_import"] * co2_grid + result_df["gas_boiler_heat"] * co2_gas
    result_df["co2_cost"] = result_df["co2_emissions"] * co2_price
    result_df["total_cost"] = result_df["cost_grid"] + result_df["co2_cost"]
    all_results.append(result_df)

# Save final output
final_df = pd.concat(all_results)
final_df.to_csv("final_hourly_simulation_cost_expected.csv", index=False)
