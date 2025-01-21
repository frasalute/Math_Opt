import math
import pandas as pd
from gurobipy import Model, GRB, quicksum
import networkx as nx
import matplotlib.pyplot as plt

file_path = '/Users/francescasalute/Dropbox/Mac/Documents/Master in Data Science/Third Semester/Mathematical Opt/Exam/Math_Opt/MO_Q1_Wind_Farm_Cable.csv'
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
print(data.columns)

# Extract node coordinates and power production
nodes = {
    row['ID']: (row['X-Coordinate (m)'], row['Y-Coordinate (m)'])
    for _, row in data.iterrows()
}

# Identify turbines and the substation
turbines = [row['ID'] for _, row in data.iterrows() if row['Point Type'].strip() == 'Turbine']
substation = [row['ID'] for _, row in data.iterrows() if row['Point Type'].strip() == 'Subsection'][0]

# Power production for each node
production = {
    row['ID']: row['Power Production (MW)']
    for _, row in data.iterrows()
}

# Euclidean distance
def euclidean_distance(i, j):
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

# Distance
distances = {
    (i, j): euclidean_distance(i, j)
    for i in nodes for j in nodes if i != j
}

#k = 5 
u = 393  
k_values = range(1, 101)  # Testing k values from 2 to 100
results = []
for k in k_values:
    print(f"Testing k = {k}")
    model = Model(f"Wind Farm Problem k={k}")

    # Decision variables
    x = model.addVars(distances.keys(), vtype=GRB.BINARY, name="x")  
    f = model.addVars(distances.keys(), lb=0, vtype=GRB.CONTINUOUS, name="f")  

    # Objective function
    model.setObjective(
        quicksum(u * distances[i, j] * x[i, j] for i, j in distances),
        GRB.MINIMIZE
    )

    # Constraints

    for i in turbines:
        model.addConstr(
            quicksum(f[i, j] for j in nodes if j != i) - quicksum(f[h, i] for h in nodes if h != i) == production[i],
            name=f"FlowConservation_Turbine_{i}"
        )

    model.addConstr(
        quicksum(f[substation, j] for j in nodes if j != substation) == 0,
        name="NoOutflow_Substation"
    )
    model.addConstr(
        quicksum(f[h, substation] for h in nodes if h != substation) == sum(production[i] for i in turbines),
        name="Inflow_Substation"
    )

    for i, j in distances:
        model.addConstr(f[i, j] <= k * x[i, j], name=f"Capacity_{i}_{j}")

    for i in turbines:
        model.addConstr(
            quicksum(x[i, j] for j in nodes if j != i) == 1,
            name=f"Outdegree_Turbine_{i}"
        )

    model.addConstr(
        quicksum(x[substation, j] for j in nodes if j != substation) == 0,
        name="NoOutdegree_Substation"
    )

    '''model.addConstr(
        quicksum(x[i, substation] for i in nodes if i != substation) <= 13, 
        name="Max3Cables_Substation")'''

    # Solve the model
    model.optimize()

    # Store results
    if model.status == GRB.OPTIMAL:
            results.append((k, model.objVal))
            print(f"Total cost for k = {k}: {model.objVal}")
    else:
            results.append((k, None))
            print(f"No solution for k = {k}")
    
    # Summarize results
    stable_k_start = None
    unique_costs = set()
    print("\nSummary of Results:")
    for k, cost in results:
        if cost is not None:
            print(f"k = {k}, Total Cost = {cost:.2f}")
            unique_costs.add(cost)
            if stable_k_start is None and len(unique_costs) == 1:
                stable_k_start = k

    print(f"\nNumber of unique total costs: {len(unique_costs)}")
    print(f"Cost stabilizes at k >= {stable_k_start} with total cost = {cost:.2f}")

# Plot results
k_values, total_costs = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(k_values, total_costs, marker='o', linestyle='-', label="Total Cost")
plt.title("Impact of Cable Capacity (k > 1) on Total Cost", fontsize=16)
plt.xlabel("Cable Capacity (k)", fontsize=14)
plt.ylabel("Total Cost (USD)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()


'''# The results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Total cost: {model.objVal}")
    print("Cables placed:")
    
    G = nx.DiGraph()
    for node_id, (x_coord, y_coord) in nodes.items():
        G.add_node(node_id, pos=(x_coord, y_coord))
    
    edges = []
    
    for i, j in distances:
        if x[i, j].x > 0.5:  # Binary variable
            print(f"  Cable from {i} to {j}, Flow: {f[i, j].x:.2f} MW")
            edges.append((i, j, f[i, j].x))  

    for i, j, flow in edges:
        G.add_edge(i, j, weight=flow)

    # Visualization
    pos = nx.get_node_attributes(G, 'pos')  
    edge_labels = nx.get_edge_attributes(G, 'weight')  

    plt.figure(figsize=(12, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(i, j): f"{flow:.2f} MW" for i, j, flow in edges}, font_size=8
    )
    
    plt.title("Wind Farm Cable Network", fontsize=16)

    # Save the plot as a PNG file
    output_file = "wind_farm_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

    plt.show()

else:
    print("No optimal solution found.")'''


