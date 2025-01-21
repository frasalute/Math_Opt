# Wind Farm Cable Optimization and Dissimilarity Clustering

This repository contains two optimization problems: the **Wind Farm Cable Network Optimization** and the **Dissimilarity Clustering** problem. Both projects utilize mathematical optimization techniques to solve real-world challenges.

---

## 1. Wind Farm Cable Optimization

### Description
This project models the optimization of a wind farm's cable network. The goal is to minimize the total cost of connecting turbines to a substation while satisfying energy flow and cable capacity constraints.

### Features
- **Euclidean Distance Calculation**: Computes distances between nodes (turbines and substations).
- **Energy Flow Optimization**: Ensures energy flow conservation and optimizes the cable layout.
- **Parameter Testing**: Varies cable capacity (`k`) to analyze cost changes and determine stability.
- **Visualization**:
  - Plots the cost trend for different `k` values.
  - Graphically represents the wind farm cable network.

### Setup
#### Install Dependencies:
```bash
pip install math gurobipy pandas matplotlib networkx
```

#### Run the Script:
```bash
python wind_farm_optimization.py
```

#### Input File:
Place `MO_Q1_Wind_Farm_Cable.csv` in the appropriate directory.

### Outputs
- **Cost Analysis**: Logs costs for varying `k` values.
- **Plots**:
  - Cost vs. Cable Capacity
  - Wind Farm Cable Network

---

## 2. Dissimilarity Clustering

### Description
This project clusters data points based on a provided dissimilarity matrix. It minimizes intra-cluster dissimilarities using a Mixed Integer Quadratic Programming (MIQP) approach.

### Features
- **Cluster Assignments**: Assigns points to clusters while minimizing dissimilarity.
- **Custom Constraints** (optional):
  - **Must-link Constraints**: Forces specific points to be in the same cluster (commented out in the code).
  - **Cannot-link Constraints**: Prevents specific points from sharing a cluster (commented out in the code).

### Setup
#### Install Dependencies:
```bash
pip install gurobipy ast
```

#### Run the Script:
```bash
python dissimilarity_clustering.py
```

#### Input File:
Place `MO_Q2_DissimilarityMatrix.txt` in the appropriate directory.

### Outputs
- **Optimal Objective Value**: Displays the minimized dissimilarity.
- **Cluster Assignments**: Maps each point to its respective cluster.

---

## Dependencies
- **Gurobi**: Optimization solver ([installation guide](https://www.gurobi.com/downloads/)).
- Python libraries:
  - `pandas`
  - `matplotlib`
  - `networkx`
  - `ast`

---

## Visualization Examples

### Cost vs. Cable Capacity
Displays the impact of varying cable capacity (`k`) on total cost.

### Wind Farm Cable Network
Illustrates the optimized cable layout, including flow values.

---

## Notes
1. Ensure the required input files are available in the specified paths.
2. Gurobi must be properly installed and licensed to run the scripts.
3. For further details, refer to the comments within each script or feel free to contact me.
