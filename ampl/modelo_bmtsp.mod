# Problema Acotado del Vendedor Viajero Múltiple (BMTSP)
# Modelo AMPL

param n;                   # número de ciudades (sin depósito)
param k;                   # número de vendedores
param max_ciudades;        # máximo de ciudades por vendedor

set N := 1..n;             # conjunto de ciudades excluyendo el depósito
set V := 0..n;             # conjunto de nodos incluyendo el depósito 0
set S := 1..k;             # conjunto de vendedores

param dist{V,V} >= 0;      # matriz de distancias simétrica

# Variables binarias: 1 si el vendedor s viaja de i a j
var x{S,V,V} binary;

# Variables MTZ para eliminar subtours para cada vendedor
var u{S,N} >= 1 <= n;

# Objetivo: minimizar la distancia total
minimize Distancia_Total:
    sum{s in S, i in V, j in V: i <> j} dist[i,j] * x[s,i,j];

# Cada ciudad tiene exactamente una arista entrante y una saliente en total
s.t. UnaEntrada{j in N}:
    sum{s in S, i in V: i <> j} x[s,i,j] = 1;

s.t. UnaSalida{i in N}:
    sum{s in S, j in V: i <> j} x[s,i,j] = 1;

# Conservación de flujo para cada vendedor en nodos que no son depósito
s.t. Flujo{s in S, i in N}:
    sum{j in V: j <> i} x[s,i,j] - sum{j in V: j <> i} x[s,j,i] = 0;

# Cada vendedor sale y entra al depósito exactamente una vez
s.t. InicioDeposito{s in S}:
    sum{j in N} x[s,0,j] = 1;

s.t. FinDeposito{s in S}:
    sum{i in N} x[s,i,0] = 1;

# Límite de ciudades para cada vendedor (número total de ciudades distintas que visita)
s.t. LimiteCiudades{s in S}:
    sum{i in N, j in V: i != j} x[s,i,j] <= max_ciudades;

# Eliminación de subtours MTZ
s.t. MTZ{s in S, i in N, j in N: i <> j}:
    u[s,i] - u[s,j] + (n) * x[s,i,j] <= n - 1;
# El depósito tiene orden 0 implícitamente; no está modelado explícitamente

