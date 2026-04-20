# 🚁 Drone Flight Physics Simulator + Research Platform

> Interactive UAV design simulator combining physics, battery systems, optimization, and data analytics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Plotly](https://img.shields.io/badge/Charts-Plotly-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

This project is a **computational drone engineering platform** that allows users to simulate, analyze, and optimize drone flight performance.

Instead of physically building drones, this system lets users test thousands of configurations digitally.

Users can adjust:

- Weight
- Motor strength
- Propeller efficiency
- Battery size
- Throttle level

Then simulate:

- Flight trajectory
- Battery consumption
- Stability
- Flight time
- Design efficiency

---

# 🎯 Why This Project Matters

This project combines multiple engineering disciplines:

### Mechanical Engineering
- Newtonian motion
- Force balance
- Flight dynamics

### Electrical Engineering
- Battery systems
- Power consumption
- Motor modeling

### Computer Science
- Simulation engine
- Optimization algorithms
- Interactive UI

### Data Science
- Parameter sweeps
- Sensitivity analysis
- Pareto optimization

---

# 🧠 Core Physics Model

### Net Force

```math id="l5f9uv"
F_{net} = T - mg