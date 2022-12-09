import numpy as np
import random as rd

exemplo = np.array([[np.inf, 3, 1, 5, 8],
                  [3, np.inf, 6, 7, 9],
                  [1, 6, np.inf, 4, 2],
                  [5, 7, 4, np.inf, 3],
                  [8, 9, 2, 3, np.inf]])

def Manhattan(x, y):
    matriz = np.full((len(x), len(y)), 0, dtype=int)
    for i in range(len(x)):
        for j  in range(len(y)):
            if (i == j):
                matriz[i][j] = np.inf
            else:
                dist = abs(x[i] - x[j]) + abs(y[i] - y[j])
        matriz[i][j] = dist
        matriz[j][i] = dist
    return matriz

def Euclides(x, y):
    matriz = np.full((len(x), len(y)), 0, dtype=int)
    for i in range(len(x)):
        for j  in range(len(y)):
            if (i == j):
                matriz[i][j] = np.inf
            else:
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) 
                dist = abs(dist)
        matriz[i][j] = dist
        matriz[j][i] = dist
    return matriz

def geradorPontos(N):
    xCoordenadas = np.random.random_integers(0, 400, size = N)
    yCoordenadas = np.random.random_integers(0, 400, size = N)
    x = np.unique(xCoordenadas)
    while (len(xCoordenadas) < N):
        x = np.append(x, np.random.random_integers(0, 40))
        x = np.unique(xCoordenadas)
    return xCoordenadas, yCoordenadas