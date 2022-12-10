from networkx.algorithms import tree
import networkx as nx
import numpy as np
import numpy as np
import heapq as heap
import sys
import time
import signal
import csv

instancias = sys.argv[1]
tipoDistancia = sys.argv[2]
algoritmo = sys.argv[3]
instancias = int(instancias)

def Manhattan(x, y):
    matriz = np.full((len(x), len(y)), 0, dtype=float)
    for i in range(len(x)):
        for j  in range(len(y)):
            if (i == j):
                dist = np.inf
            else:
                dist = abs(x[i] - x[j]) + abs(y[i] - y[j])
            matriz[i][j] = dist
            matriz[j][i] = dist
    return matriz

def Euclides(x, y):
    matriz = np.full((len(x), len(y)), 0, dtype=float)
    for i in range(len(x)):
        for j  in range(len(y)):
            if (i == j):
                dist = np.inf
            else:
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) 
                dist = abs(dist)
            matriz[i][j] = dist
            matriz[j][i] = dist
    return matriz

def geradorPontos(N):
    xCoordenadas = np.random.randint(0, 400, size = N)
    yCoordenadas = np.random.randint(0, 400, size = N)
    x = np.unique(xCoordenadas)
    while (len(xCoordenadas) < N):
        x = np.append(x, np.random.random_integers(0, 40))
        x = np.unique(xCoordenadas)
    return xCoordenadas, yCoordenadas

def definindoInstancias(instancias, tipoDistancia):
    x, y  = geradorPontos(instancias)
    matrizDistancia = []
    if (tipoDistancia == 'euclidiana'):
        matrizDistancia = Euclides(x, y)
    elif(tipoDistancia == 'manhattan'):
        matrizDistancia = Manhattan(x, y)
    G = nx.complete_graph(instancias)
    for (u,v) in G.edges():
        G.edges()[u,v]['weight'] = matrizDistancia[u][v]
        
    return matrizDistancia, G

class Node:
    def __init__(self, bounds, nivel, custo, solucao):
        self.bounds = bounds
        self.nivel = nivel
        self.custo = custo
        self.solucao = solucao
        
    def __lt__(self, other):
        if(self.bounds < other.bounds):
            return True
        else:
            if(self.bounds == other.bounds):
                if(self.nivel > other.nivel):
                    return True
                else:
                    if(self.solucao[self.nivel] < other.solucao[other.nivel]):
                        return True
        return False

def bound(matriz, solucao):
    soma = 0
    tamanho = len(matriz[0][:])
    resto = []
    for i in range(tamanho):
        if i not in solucao:
            resto.append(i)
    for i in range(len(solucao) - 1):
        soma += matriz[solucao[i]][solucao[i + 1]]
    min = np.inf
    for i in resto:
        if(matriz[solucao[-1]][i] < min):
            min = matriz[solucao[-1]][i]
    if(len(resto) != 0):
        soma += min
    p = [solucao[0]] + resto
    for r in resto:
        min = np.inf
        for i in p:
            if(i != r):
                if(matriz[r][i] < min):
                    min = matriz[r][i]
        soma += min
    return soma * 2

def boundFinal(matriz, solucao):
    soma = 0
    for i in range(len(solucao) - 1):
        soma += matriz[solucao[i]][solucao[i + 1]]
    return np.ceil(soma / 2)
        
def bnbTsp(A, n):
    raiz = Node(bound(A, [0]), 0, 0, [0])
    fila = []
    heap.heapify(fila)
    heap.heappush(fila, raiz)
    melhorValor = np.inf
    melhorSolucao = []
    while (len(fila) != 0):
        no = heap.heappop(fila)
        if (no.nivel > n - 1):
            if (melhorValor > no.custo):
                melhorValor = no.custo
                melhorSolucao = no.solucao
        else:
            if (no.bounds < melhorValor):
                if(no.nivel < n - 1):
                    for k in range(n):
                        if (k not in no.solucao) and (A[no.solucao[-1]][k] != np.inf) and (bound(A, np.append(no.solucao, k)) < melhorValor):
                            novoNo = Node(bound(A, np.append(no.solucao, k)), no.nivel + 1, no.custo + A[no.solucao[-1]][k], np.append(no.solucao, k))
                            heap.heappush(fila, novoNo)
                else:
                    if (A[no.solucao[-1]][0] != np.inf and boundFinal(A, np.append(no.solucao, 0)) < melhorValor):
                        novoNo = Node(boundFinal(A, np.append(no.solucao, 0)), no.nivel + 1, no.custo + A[no.solucao[-1]][0], np.append(no.solucao, 0))
                        heap.heappush(fila, novoNo)
                        
    return melhorSolucao, melhorValor

def approxTspTour(G, A, c):
    arvoreMinima = tree.minimum_spanning_tree(G,algorithm="prim")
    caminhamentoPreOrdem = nx.dfs_preorder_nodes(arvoreMinima, source=0)
    cicloHamiltoniano = list(caminhamentoPreOrdem) + [0]
    custo = boundFinal(A, cicloHamiltoniano)
    return cicloHamiltoniano

def approxTspChristofides(G, A, c):
    arvoreMinima = tree.minimum_spanning_tree(G,algorithm="prim")
    verticesImpares = []
    for i in range(c):
        if(arvoreMinima.degree()[i] % 2 == 1):
            verticesImpares.append(i)
    grafoInduzido = nx.Graph(G.subgraph(verticesImpares))
    matchMinimo = nx.min_weight_matching(grafoInduzido)
    multigrafo = nx.MultiGraph(arvoreMinima)
    for (u,v) in matchMinimo:
        tripla = (u, v, A[u][v])
        multigrafo.add_weighted_edges_from([tripla])
    circuitoEuliriano = nx.eulerian_circuit(multigrafo)
    solucao = []
    visitados = [False] * c
    for u,v in circuitoEuliriano:
        if(visitados[u] == False):
            solucao += [u]
            visitados[u] = True
    solucao += [0]
    return solucao

def signal_handler(signum,frame):
    raise Exception("Timed out!")

signal.signal(signal.SIGALRM, signal_handler)
signal.alarm(1800)

tests = open('tests.csv','a')

writer = csv.writer(tests)

A, G = definindoInstancias(instancias, tipoDistancia)

inicio = 0.0
fim = 0.0

try:
    if(algoritmo == 'bnb'):
        inicio = time.time()
        solucao, custo = bnbTsp(A, instancias)
        fim = time.time()
    elif(algoritmo == 'tour'):
        solucao = approxTspTour(G, A, 2**(instancias))
    elif(algoritmo == 'chr'):
        solucao = approxTspChristofides(G, A,  2**(instancias))
    print(fim - inicio)
    writer.writerow(solucao)
    print(custo)
    tests.close()   
except Exception:
    tests.close()    