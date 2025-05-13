# ICODE: Interpretable Causality ODE Networks

This repository contains the source code and datasets for **ICODE**, a model designed for explainable anomaly detection via interpretable dynamical systems.

---

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yus516/ICODE.git
cd ICODE
```

2. **Set up the Conda environment:**

```bash
conda env create -f environment.yml
conda activate ICODE
```

---

## 📊 Data Generation

Run the following scripts to generate or load datasets:

### 🔁 Simulated Anomaly Data

This includes:
- **Lorenz 96**
- **Lotka–Volterra**
- **Reaction–Diffusion**

📌 Note:
- If a dataset file is not detected, it will be simulated automatically.
- If found, it will be loaded instead.
- Measurement anomalies are added via `add_measurement_anomaly`.
- Cyber anomalies are introduced using `simulate_with_cyber_anomaly`.

To train the Neural ODE model and infer causal graphs:

```bash
bash run_grid_search_lorenz96
```

#### Parameters Description
-`num-sim`:             # Number of simulation
-`K`:                   # Used only in lorenz 96 system
-`num-hidden-layers`:   # Number of hidden layer
-`hidden-layer-size`:   # Number of units per hidden layer
-`batch-size`:          # Batch size used during training
-`num-epochs`:          # Number of training epochs
-`initial-lr`:          # Initial learning rate for optimization
-`seed`:                # Random seed for reproducibility

### ⚡ Power System Anomaly Data

For simulating power system anomalies, refer to [PNNL GridSTAGE](https://github.com/pnnl/GridSTAGE).

To execute training and evaluation:

```bash
bash run_grid_search_power
```

📁 The adjacency matrix output file is defined in line 111 of `run_grid_search.py`.  
📂 All outputs will be saved in the `tmp` directory.

---

## 📈 Running Analysis

To evaluate results and get the type and root cause, use (including identifying measurement and cyber anomalies with root causes):
```bash
python check_measure_cyber.py
```

For the power system, use

```bash
python analysis.py
```

🧠 Additional analysis and visualization are included in `show-dynamical-causal.ipynb`.

---
