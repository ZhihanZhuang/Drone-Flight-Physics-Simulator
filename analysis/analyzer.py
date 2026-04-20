import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

def run_analysis(df):
    """
    Processes the raw dataset to extract correlations, find the Pareto frontier,
    and cluster drone types. Generates standard research visualizations.
    """
    os.makedirs("results_plots", exist_ok=True)
    
    # Filter only drones that could lift off
    feasible_df = df[df['feasible'] == 1].copy()
    if feasible_df.empty:
        print("Warning: No feasible designs found in simulation.")
        return

    # 1. Correlation Analysis
    corr = feasible_df[['mass', 'motor_k', 'battery', 'flight_time', 'energy_efficiency', 'stability_score']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Drone Parameter Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("results_plots/correlation_matrix.png")
    plt.close()

    # 2. Pareto Frontier (Objective: Maximize Flight Time, Minimize Mass)
    costs = feasible_df[['mass', 'flight_time']].values
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    
    for i, c in enumerate(costs):
        for j, other in enumerate(costs):
            if i != j:
                # 'other' dominates 'c' if it is lighter AND flies longer
                if other[0] <= c[0] and other[1] >= c[1] and (other[0] < c[0] or other[1] > c[1]):
                    is_pareto[i] = False
                    break

    pareto_df = feasible_df[is_pareto]

    plt.figure(figsize=(9, 6))
    plt.scatter(feasible_df['mass'], feasible_df['flight_time'], label='Sub-optimal Designs', alpha=0.4, c='gray')
    plt.scatter(pareto_df['mass'], pareto_df['flight_time'], color='red', s=60, label='Pareto Optimal Frontier')
    plt.xlabel("Mass (kg) -> lower is better")
    plt.ylabel("Flight Time (s) -> higher is better")
    plt.title("Optimization: Pareto Frontier of Drone Designs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig("results_plots/pareto_frontier.png")
    plt.close()

    # 3. K-Means Clustering Analysis
    # Grouping designs to discover archetypes (e.g., Heavy Lifters, Agile, Long Endurance)
    features = feasible_df[['mass', 'motor_k', 'battery']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    feasible_df['cluster'] = kmeans.fit_predict(features)
    
    cluster_labels = {0: "Archetype A", 1: "Archetype B", 2: "Archetype C"}
    feasible_df['Cluster Name'] = feasible_df['cluster'].map(cluster_labels)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=feasible_df, x='battery', y='flight_time', hue='Cluster Name', palette='viridis', s=80)
    plt.title("Drone Archetype Clustering (K-Means)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results_plots/design_clusters.png")
    plt.close()