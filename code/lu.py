import numpy as np

A = np.array([[3,18,-6],[-5,-23,31],[9,65,13]])
b = np.array([9,7,2])

A1 = np.array([[1,5,3,4],[1,8,15,16],[1,12,33,42],[1,9,24,48]]) 
b1 = np.array([29,95,217,211])

def LU(A, b):
    n = A.shape[0]
    ones = np.ones(n)
    L = np.zeros((n,n))
    U = np.diag(ones)

    for i in range(n):
        for j in range(i+1):
            scal_prod = 0
            for k in range(j):
                scal_prod += L[i,k] * U[k,j]
            L[i,j] = A[i,j] - scal_prod
        
        for j in range(i+1,n):
            scal_prod = 0
            for k in range(i):
                scal_prod += L[i,k] * U[k,j]
            U[i,j] = (A[i,j] - scal_prod) / L[i,i]

    print("b:\n", b, "\n")
    print("A:\n", A, "\n")
    print("L:\n", L, "\n")
    print("U:\n", U, "\n")

    # Solve (1) Ly = b and (2) Ux = y

    y = np.zeros(n)
    y[0] = b[0] / L[0,0]
    for i in range(1, n):
        scal_prod = 0
        for k in range(i):
            scal_prod += L[i,k] * y[k]
        y[i] = (b[i] - scal_prod) / L[i,i]

    print("Intermediate Solution 'y': ", y, "\n")

    x = np.zeros(n)
    x[n-1] = y[n-1] #U[n-1, n-1] = 1
    for i in range(n-2,-1,-1):
        scal_prod = 0
        for k in range(i+1, n):
            scal_prod += U[i,k] * x[k]
        x[i] = y[i] - scal_prod # U[i,i] = 1

    print("Final Solution 'x': ", x)

    print("Ax:\n", np.matmul(A,x), "\nb\n", b)

LU(A1, b1)
