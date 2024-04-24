import numpy as np

A = np.array([[9,-3,1],[-3,2,0],[1,0,4]])
b = np.array([5,3,2])

A1 = np.array([[1,2,5,7],[2,5,15,17],[5,15,54,56],[7,17,56,68]])
b1 = np.array([46,123,419,469])

def solve_system_via_cholesky(A, b):
    U = cholesky(A)
    y = solve_lower_traingular_system(U.T, b)
    x = solve_upper_traingular_system(U, y)
    print("\nAx:\n", np.matmul(A,x), "\nb\n", b)

def cholesky(A):
    n = A.shape[0]
    U = np.zeros((n,n))

    for i in range(n):
        scal_prod = 0
        for k in range(i):
            scal_prod += U[k,i] ** 2
        U[i,i] = np.sqrt(A[i,i] - scal_prod)
        for j in range(i+1, n):
            scal_prod = 0
            for k in range(i):
                scal_prod += U[k,i]*U[k,j]
            U[i,j] = (A[i,j] - scal_prod) / U[i,i]
    
    print(U)
    return U

def solve_lower_traingular_system(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0] / L[0,0]
    for i in range(1, n):
        scal_prod = 0
        for k in range(i):
            scal_prod += L[i,k] * x[k]
        x[i] = (b[i] - scal_prod) / L[i,i]

    print("Intermediate Solution 'x': ", x, "\n")
    return x

def solve_upper_traingular_system(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    x[n-1] = b[n-1] / U[n-1, n-1]
    for i in range(n-2,-1,-1):
        scal_prod = 0
        for k in range(i+1, n):
            scal_prod += U[i,k] * x[k]
        x[i] = (b[i] - scal_prod) / U[i,i] 

    print("Final Solution 'x': ", x)
    
    return x

solve_system_via_cholesky(A1, b1)