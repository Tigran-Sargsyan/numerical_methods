import numpy as np

A = np.array([[3,18,-6],[-5,-23,31],[9,65,13]])
b = np.array([9,7,2])

A1 = np.array([[1,5,3,4],[1,8,15,16],[1,12,33,42],[1,9,24,48]]) 
b1 = np.array([29,95,217,211])

def solve_system_via_lu(A, b):
    L, U = LU(A)
    y = solve_lower_traingular_system(L, b)
    x = solve_upper_traingular_system(U, y)
    print("\nAx:\n", np.matmul(A,x), "\nb\n", b)

def LU(A):
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

    print("A:\n", A, "\n")
    print("L:\n", L, "\n")
    print("U:\n", U, "\n")

    return L, U

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

solve_system_via_lu(A, b)