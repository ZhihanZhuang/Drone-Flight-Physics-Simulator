import os
from experiments.batch_simulator import run_grid_search
from analysis.analyzer import run_analysis
from ml_model.train_model import train_rf_model

def setup_directories():
    """Initializes the required project folder architecture"""
    dirs = [
        'core_sim', 'experiments', 'data/raw_runs', 
        'data/processed', 'analysis', 'optimization', 
        'ml_model/saved', 'notebooks', 'results_plots'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    print("="*50)
    print("🚁 DRONE FLIGHT PHYSICS & RESEARCH PLATFORM")
    print("="*50)
    
    setup_directories()

    print("\n[Phase 1 & 2] Running Physical Simulation Batch...")
    dataset = run_grid_search()
    print(f"-> Generated dataset with {len(dataset)} design configurations.")

    print("\n[Phase 3 & 4] Conducting Data Analysis & Pareto Optimization...")
    run_analysis(dataset)
    print("-> Research plots generated in 'results_plots/' directory.")

    print("\n[Phase 5] Training Machine Learning Predictive Model...")
    train_rf_model(dataset)

    print("\n🚀 Run Complete! Workflow finalized.")
    print("Next steps: Review 'results_plots/pareto_frontier.png' for optimal designs.")