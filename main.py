import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix



# I will use dataset which has been processed into a matrix 
dataPath = "./data/COIL20.mat"

dataSize = 1440
featureNum = 1024
nonZeroValueinAmatrix = 11
classNum = 20 # the number of class
lambdas = 0.25
eps = 1e-13


def dataLoader(dataPath):
    data = loadmat(dataPath)
    X = np.array(data['X'], dtype=np.float32)
    label = np.array(data['Y'])

    assert dataSize == X.shape[0]
    assert featureNum == X.shape[1]

    print(X.shape)
    return X, label

def L2_distance(X1, X2):
    X1_squared = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_squared = np.sum(X2 ** 2, axis=1, keepdims=True)
    return X1_squared + X2_squared.T - 2 * np.dot(X1, X2.T)

# construct similarity matrix from data
# and denote it as A
def computeWeightedAdjacency(X, k = nonZeroValueinAmatrix):
    num = X.shape[0]
    
    # Step 1: 计算距离矩阵
    distX = L2_distance(X, X)  # 计算欧几里得距离平方
    
    # Step 2: 排序并获取索引
    idx = np.argsort(distX, axis=1)  # 对每一行排序，返回索引
    distX_sorted = np.sort(distX, axis=1)  # 对每一行排序，返回值
    
    # Step 3: 初始化矩阵
    A = np.zeros((num, num))
    rr = np.zeros(num)
    
    # Step 4: 遍历每个点，计算权重
    for i in range(num):
        di = distX_sorted[i, 1:k+2]  # 取前 k+2 个距离（排除自身距离）
        rr[i] = 0.5 * (k * di[-1] - np.sum(di[:-1]))
        id_neighbors = idx[i, 1:k+2]  # 对应的索引
        A[i, id_neighbors] = (di[-1] - di) / (k * di[-1] - np.sum(di[:-1]) + np.finfo(float).eps)
    
    return (A + A.T) / 2


# F is continuous indcator
# use eigenvalue decomposition
def initializeF(A):
    LaplacianA = np.diag(np.sum(A, axis = 1)) - A

    eValue, eVector = np.linalg.eigh(LaplacianA)
    return eVector[:, :classNum]

def updateF(LaplacianA, LaplacianS, lambdas):
    sumLaplacian = LaplacianA + lambdas * LaplacianS
    eValue, eVector = np.linalg.eigh(sumLaplacian)

    return eVector[:, :classNum]

def NewtonMethodtoFindOptimalLambda(u):
    eps = 1e-13
    epoch = 1000
    def f(nowLambda, u):
        return np.sum(np.maximum(nowLambda - u, 0)) / u.size - nowLambda
    
    def derivationF(nowLamdbda, u):
        return np.count_nonzero(u[u < nowLamdbda]) / u.size - 1
     
    len = np.max(u) - np.min(u)
    derivValue = 0
    
    # find a initial optimalLambda such that derivValue not equals zero
    while derivValue == 0:
        optimalLambda = np.random.normal(loc= len / 2 + np.min(u)
                                     , scale= len / 8)
        derivValue = derivationF(optimalLambda, u)
    
    for i in range(epoch):
        derivValue = derivationF(optimalLambda, u)
        assert derivValue != 0
        valueOfFLambda = f(optimalLambda, u)

        optimalLambda = optimalLambda - valueOfFLambda\
                        / derivValue
        if np.abs(valueOfFLambda) < eps:
            break;

    assert optimalLambda >= 0

    return optimalLambda

def testEqual(a, b):
    eps = 1e-10
    return np.abs(a - b) < eps

def updateS(F, lambdas, alphaList):
    V = np.zeros((dataSize, dataSize))
    S = np.zeros_like(V)
    
    V = np.sum(F ** 2, axis = 1).reshape(-1, 1) + \
            np.sum(F ** 2, axis = 1) - 2 * np.dot(F, F.T)
    
    alpha = alphaList[0]
    if alpha == 0:
        sortedV = np.sort(V, axis = 1)
        k = nonZeroValueinAmatrix
        for i in range(dataSize):
            alpha += lambdas * k * sortedV[i, k] / 4
            for j in range(k):
                alpha -= lambdas / 4 * sortedV[i,j]
        alpha /= dataSize
        alphaList[0] = alpha
        print("alpha = ", alpha)
    
    V = -lambdas / (4 * alpha) * V

    U = np.zeros_like(V)
    for i in range(dataSize):
        U[i] = V[i] - np.sum(V[i]) / dataSize + \
                1.0 / dataSize
        optimalLambda = NewtonMethodtoFindOptimalLambda(U[i])
        S[i] = np.maximum(U[i] - optimalLambda, 0)

        # verify KKT conditions
        # gamma = (1.0 - np.sum(V[i])) / dataSize - optimalLambda
        # for j in range(dataSize):
        #     lambdaj = S[i, j] - U[i, j] + optimalLambda
        #     if np.abs(lambdaj) < eps :
        #         lambdaj = 0
        #     # print(lambdaj)
        #     assert testEqual(S[i, j] - V[i, j] - gamma - lambdaj, 0)
        #     assert S[i, j] >= 0
        #     assert lambdaj >= 0, print(lambdaj)
        #     assert testEqual(S[i, j] * lambdaj, 0)
        # assert testEqual(np.sum(S[i]), 1)

    return S;

def optimization(A, lambdas):
    S = np.copy(A)
    epoch = 40
    F = initializeF(A)
    LaplacianA = np.diag(np.sum(A, axis = 1)) - A
    alphaList = [0]
    for i in range(epoch):
        S = updateS(F, lambdas, alphaList)
        symmetricS = (S.T + S) / 2
        LaplacianS = np.diag(np.sum(symmetricS, axis = 1))- symmetricS
        F = updateF(LaplacianA, LaplacianS, lambdas)

        if i % 10 == 0:
            print("The epoch: ", i)
            print(lambdas * np.trace(F.T @ LaplacianS @ F))
            print(np.trace(F.T @ LaplacianA @ F) + alphaList[0] * np.sum(S ** 2))
    
    return S, F

def clusteringAccuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / y_pred.size

X, label = dataLoader(dataPath)

A = computeWeightedAdjacency(X)

Y = initializeF(A)

S, F = optimization(A, lambdas)

symmetricS = (S.T + S) / 2

kmeans = KMeans(n_clusters = 20)
predictLable = kmeans.fit_predict(F)

# verify constraint
print("constraint row ", np.sort(np.sum(S, axis = 1)))
print("minimum of S", np.min(S))
I = np.diag(np.ones(classNum))
print("is F^T @ F an identity matrix? ", np.all(np.abs(F.T @ F - I) < 1e-7))

symmetricS[symmetricS < eps] = 0

# convert to sparse matrix
graph = csr_matrix(symmetricS)

# compute connected component
components, labelS = connected_components(csgraph=graph, directed=False)

print("number of components: ", components)

print("Acc from F: ", clusteringAccuracy(label-1, predictLable) * 100)
print("Acc from S: ", clusteringAccuracy(label-1, labelS) * 100)
