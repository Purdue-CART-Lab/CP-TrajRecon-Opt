# Integrated optimization for vehicle trajectory reconstruction under cooperative perception environment

This repository contains the source code accompanying the paper  
**“Integrated Optimization for Vehicle Trajectory Reconstruction under Cooperative Perception Environment”**,  
published in *Transportation Research Part C: Emerging Technologies*  
(https://doi.org/10.1016/j.trc.2026.105522).

The proposed vehicle trajectory reconstruction pipeline consists of **three main components**:

1. **Data Preparation**: This module first **uniformly samples Connected and Automated Vehicles (CAVs)** according to a predefined **market penetration rate (MPR)** over the selected trajectory dataset. It then generates CAV observations using one of the following perception models:

    - a **distance-dependent True Positive Rate (TPR)** model, or  
    - an **occlusion-aware detection model** from https://doi.org/10.1080/15472450.2024.2307031.

    Finally, the processed trajectories are organized into a dictionary covering all planning horizons and saved as a `.pkl` file in the `./data` directory.

2. **Optimization**: This module takes the generated `.pkl` file as input and formulates a **Mixed-Integer Linear Programming (MILP)** problem to reconstruct full vehicle trajectories from partial CAV observations.

    The reconstruction results are exported as:
    - **CSV files** (numerical results), and
    - **figures** (trajectory visualizations),

    all of which are saved in the `./results` directory.

3. **Evaluation**: Based on the reconstructed trajectory CSV files, this module evaluates reconstruction performance using **five metrics**:

    - **MAE_x**: Mean Absolute Error of longitudinal position  
    - **MAPE_x**: Mean Absolute Percentage Error of longitudinal position  
    - **RMSE_x**: Root Mean Square Error of longitudinal position  
    - **MAE_k**: Mean Absolute Error of lane index  
    - **MAE_LC**: Mean Absolute Error of lane-change timing  

---

> ⚠️ **IMPORTANT NOTE**  
> At the current stage, this repository is intended to **reproduce the experimental results** reported in the paper for the **NGSIM US101** and **Lankershim Boulevard** datasets.  
>  
> Applying the pipeline to **other datasets or perception models** may require modifications to the source code, such as adapting dataset-specific column names, coordinate systems, or preprocessing steps.

---

## Repository Structure

```text
CP-TrajRecon-Opt/
├─ configs/
│  ├─ dp/                      # data preparation configs (YAML)
│  └─ opt/                     # optimization configs (YAML)
├─ data/                       # raw / intermediate data (not included; see .gitignore)
├─ results/                    # optimization outputs (not included; see .gitignore)
├─ scripts/                    # runnable entry points
├─ src/                        # core modules (data prep, MILP, performance_metrics)
├─ requirements.txt
├─ requirements.lock.txt
└─ README.md
└─ LICENSE
└─ .gitignore
```

## Installation

1. **Create a Python Virtual Environment** (Python >= 3.9) using e.g., [Miniconda3](https://docs.anaconda.com/free/miniconda/):
    ```bash
    conda create -n cp-trajrecon-opt python=3.11 -y
    conda activate cp-trajrecon-opt
    ```

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/Purdue-CART-Lab/CP-TrajRecon-Opt.git
    cd CP-TrajRecon-Opt
    ```

3. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    # or, for exact replication of our experimental environment:
    pip install -r requirements.lock.txt
    ```

4. **Install MILP Solver (Gurobi)**  
   This project relies on Pyomo and requires an external MILP solver.  
   We recommend **Gurobi** for best performance.

   - Download Gurobi: https://www.gurobi.com/downloads/
   - Academic license (free for university users):  
     https://www.gurobi.com/academia/academic-program-and-licenses/

   After installing Gurobi and activating your license, install the Python interface:
   ```bash
   pip install gurobipy
   ```
   Verify the installation:
   ```bash
   python - << 'EOF'
   import gurobipy as gp
   print(gp.gurobi.version())
   EOF
   ```
> **Note:** Please ensure the Gurobi solver is accessible in your system PATH before running the optimization module.

## Usage Examples

### 1. Download raw NGSIM trajectory data

Download the following ZIP files from the USDOT NGSIM dataset page:
- **Lankershim-Boulevard-LosAngeles-CA.zip**
- **US-101-LosAngeles-CA.zip**

Extract both ZIP files. For the US-101 package, also extract the nested ZIP:
- `US-101-LosAngeles-CA/us-101-vehicle-trajectory-data.zip`

Then place the following CSVs into `./data/NGSIM/` (create the folder beforehand):
- `NGSIM__Lankershim_Vehicle_Trajectories.csv` (under `Lankershim-Boulevard-LosAngeles-CA/`)
- `trajectories-0750am-0805am.csv` (under `US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/0750am-0805am/`)

The expected layout is:
```text
data/NGSIM/
  ├── NGSIM__Lankershim_Vehicle_Trajectories.csv
  └── trajectories-0750am-0805am.csv
```

### 2. Data preparation (raw CSV -> PKL)

Process raw trajectory CSVs into PKL files for downstream optimization. Preprocessing behavior is controlled by YAML configs in **configs/dp/**. Please check **configs/dp/template.yaml** for configuration details.

Example:
```bash
python scripts/preprocessing.py --config configs/dp/us101_prob_MPR3.yaml
```

### 3. Optimization

Generate trajectory reconstruction results using MILP. Optimization behavior is controlled by YAML configs in **configs/opt/**. Please check **configs/opt/template.yaml** for configuration details.

Example:
```bash
python scripts/optimization.py --config configs/opt/us101_prob_MPR3_PLR0.yaml
```

### 4. Evaluation

Evaluate the trajecytory reconstruction results by comparing to the ground truth. The same YAML configs in **configs/opt/** are used to control the evaluation.

Example:
```bash
python scripts/evaluation.py --config configs/opt/us101_prob_MPR3_PLR0.yaml
```

## Citing This Work

If you find this work useful in your academic research, commercial products, or any published material, please consider citing our paper:

```bibtex
@article{zhu2026cptrajrecon,
  title = {Integrated optimization for vehicle trajectory reconstruction under cooperative perception environment},
  author = {Tianheng Zhu and Wangzhi Li and Yiheng Feng},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume = {184},
  pages = {105522},
  year = {2026},
  issn = {0968-090X},
  doi = {https://doi.org/10.1016/j.trc.2026.105522}
}
``` 

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/Purdue-CART-Lab/CP-TrajRecon-Opt/issues). Your contributions are greatly appreciated!

## License

This project is licensed under the MIT License, an [OSI-approved](https://opensource.org/licenses/MIT) open-source license, which allows for both academic and commercial use. By citing this project, you help support its development and acknowledge the effort that went into creating it. For more details, see the [LICENSE](LICENSE) file. Thank you!