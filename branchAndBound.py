import numpy as np
import heapq as heap

exemplo = np.array([[np.inf, 3, 1, 5, 8],
                  [3, np.inf, 6, 7, 9],
                  [1, 6, np.inf, 4, 2],
                  [5, 7, 4, np.inf, 3],
                  [8, 9, 2, 3, np.inf]])

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

def menorAresta(arestas):
    ordenado = np.sort(arestas, kind = 'quicksort')
    return ordenado[0]

def segundaMenorAresta(arestas):
    ordenado = np.sort(arestas, kind = 'quicksort')
    return ordenado[1]

def bound(matriz, solucao):
    copia = matriz.copy()
    soma = 0
    tamanho = len(matriz[0][:])
    if (len(solucao) != 1):
        for i in range(tamanho):
            if i in solucao:
                if(i == 0):
                    soma += (2 * copia[i][solucao[i + 1]])
                    copia[i][solucao[i + 1]] = np.inf
                    copia[solucao[i + 1]][i] = np.inf
                    soma += menorAresta(copia[i][:])
                else:
                    if(i < len(solucao) - 1):
                        soma += (2 * copia[i][solucao[i + 1]])
                        copia[i][solucao[i + 1]] = np.inf
                        copia[solucao[i + 1]][i] = np.inf
                    else:
                        soma += menorAresta(copia[i][:])
            else:
                soma += (menorAresta(copia[i][:]) + segundaMenorAresta(copia[i][:]))
    else:
        for i in range(len(matriz[0][:])):
            ordenado = np.sort(matriz[i][:], kind = 'quicksort')
            soma += (ordenado[0] + ordenado[1])
    return np.ceil(soma / 2)

def boundFinal(matriz, solucao):
    soma = 0
    for i in range(len(solucao) - 1):
        soma += matriz[solucao[i]][solucao[i + 1]]
    return np.ceil(soma / 2)
        
def bnbTsp(A, n):
    raiz = Node(bound(A, [0]), 0, 0, [0])
    fila = []
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
                    if (A[no.solucao[-1]][0] != np.inf and boundFinal(A, np.append(no.solucao, 0)) < melhorValor and len(no.solucao) == n):
                        novoNo = Node(boundFinal(A, np.append(no.solucao, 0)), no.nivel + 1, no.custo + A[no.solucao[-1]][0], np.append(no.solucao, 0))
                        heap.heappush(fila, novoNo)
                        
    return melhorSolucao
                        
print(bnbTsp(exemplo, 5))
    