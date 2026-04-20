import pandas as pd
import itertools
from core_sim.physics_engine import DronePhysics

def run_grid_search():
    """
    Sweeps through the drone parameter space, running the physics 
    simulation for each configuration to generate the dataset.
    """
    # Parameter Grid Space
    masses = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]
    motor_ks = [1e-5, 1.5e-5, 2.5e-5, 4e-5] 
    batteries = [1500, 3000, 5000, 8000] # mAh

    results = []

    combinations = list(itertools.product(masses, motor_ks, batteries))
    print(f"Running {len(combinations)} design configurations...")

    for m, k, b in combinations:
        sim = DronePhysics(mass=m, motor_k=k, battery_capacity_mah=b)
        
        max_alt = 0
        target_altitude = 20.0
        
        # Run simulation loop
        while sim.step(target_h=target_altitude):
            max_alt = max(max_alt, sim.h)

        # Calculate metrics
        efficiency = sim.time / sim.initial_energy if sim.initial_energy > 0 else 0
        # Stability: penalizes overshooting or failing to reach target altitude
        stability = 1.0 / (1.0 + abs(max_alt - target_altitude)) if sim.feasible else 0

        results.append({
            "mass": m,
            "motor_k": k,
            "battery": b,
            "flight_time": sim.time,
            "max_altitude": max_alt,
            "energy_efficiency": efficiency * 1000, # Scaled up for readability
            "stability_score": stability,
            "feasible": int(sim.feasible)
        })

    df = pd.DataFrame(results)
    df.to_csv("data/processed/simulation_dataset.csv", index=False)
    return df