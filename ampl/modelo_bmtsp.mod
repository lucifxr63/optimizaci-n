# Bounded Multiple Traveling Salesman Problem (BMTSP)
# AMPL model

param n;                   # number of cities (without depot)
param k;                   # number of salesmen
param max_ciudades;        # maximum cities per salesman

set N := 1..n;             # set of cities excluding depot
set V := 0..n;             # set of nodes including depot 0
set S := 1..k;             # set of salesmen

param dist{V,V} >= 0;      # symmetric distance matrix

# Binary variables: 1 if salesman s travels from i to j
var x{S,V,V} binary;

# MTZ variables to eliminate subtours for each salesman
var u{S,N} >= 1 <= n;

# Objective: minimize total distance
minimize Total_Distance:
    sum{s in S, i in V, j in V: i <> j} dist[i,j] * x[s,i,j];

# Each city has exactly one incoming and one outgoing edge overall
s.t. OneIn{j in N}:
    sum{s in S, i in V: i <> j} x[s,i,j] = 1;

s.t. OneOut{i in N}:
    sum{s in S, j in V: i <> j} x[s,i,j] = 1;

# Flow conservation for each salesman at non-depot nodes
s.t. Flow{s in S, i in N}:
    sum{j in V: j <> i} x[s,i,j] - sum{j in V: j <> i} x[s,j,i] = 0;

# Each salesman leaves and enters the depot exactly once
s.t. StartDepot{s in S}:
    sum{j in N} x[s,0,j] = 1;

s.t. EndDepot{s in S}:
    sum{i in N} x[s,i,0] = 1;

# City limit for each salesman (número total de ciudades distintas que visita)
s.t. LimitCities{s in S}:
    sum{i in N, j in V: i != j} x[s,i,j] <= max_ciudades;

# MTZ subtour elimination
s.t. MTZ{s in S, i in N, j in N: i <> j}:
    u[s,i] - u[s,j] + (n) * x[s,i,j] <= n - 1;

# Depot has order 0 implicitly; not modeled explicitly

