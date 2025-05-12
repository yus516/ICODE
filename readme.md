# ICODE

## Overview

ICODE is a framework for causal discovery using Neural Ordinary Differential Equations (Neural ODEs). It enables the simulation of measurement and cyber anomaly data and uses learned dynamics to uncover causal relationships.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yus516/ICODE.git
   cd ICODE
   ```

2. **Set up the Conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate icode
   ```

   This will install all necessary dependencies including PyTorch, NumPy, SciPy, and others.

## Data Generation

Run the following scripts to generate datasets:

- **Measurement anomaly data:**

  ```bash
  python run_grid_search_power
  ```

- **Cyber anomaly data:**

  ```bash
  python run_grid_search_lorenz96
  ```

Output files will be saved under the `datasets/` directory.

## Running Experiments

To train the Neural ODE model and perform causal discovery, execute:

```bash
python training.py
```

This script uses the generated datasets and outputs results including the learned causal graph.

## Causal Graph Output

During the run of `training.py`, the causal graph is saved. Look for lines like:
