import gurobipy as gp
from gurobipy import GRB
import ast

# Import the Dissimilarity Matrix
def read_dissimilarity_matrix(filename):
    with open(filename, 'r') as f:
        content = f.read() 
        try:
            dis = ast.literal_eval(content)
            return dis
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing dissimilarity matrix: {e}")
            return None

filename = "MO_Q2_DissimilarityMatrix.txt"
dis = read_dissimilarity_matrix(filename)

# Set Up the MIQP Model
n = len(dis)  
K = 3 # given by exercise

model = gp.Model("DissimilarityClusteringK3")

# variables
x = model.addVars(n, K, vtype=GRB.BINARY, name="x")

# obj funct
obj = gp.QuadExpr()
for i in range(n):
    for j in range(i+1, n):  
        for c in range(K):
            obj.addTerms(dis[i][j], x[i, c], x[j, c])

model.setObjective(obj, GRB.MINIMIZE)

# constraint
for i in range(n):
    model.addConstr(gp.quicksum(x[i, c] for c in range(K)) == 1)

# Optimize the model
model.optimize()

# solution
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    cluster_assignment = {}
    for i in range(n):
        for c in range(K):
            if x[i, c].X > 0.5:  
                cluster_assignment[i] = c
    print("Cluster assignments:", cluster_assignment)
else:
    print("No optimal solution found.")
