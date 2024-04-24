from math import sin,cos,e

def f(x):
    return x - e**(-cos(x))

def f_prime(x):
    return 1 - sin(x)*e**(-cos(x))

def newton_method(f, f_prime, x_curr):
    return x_curr - (f(x_curr)/f_prime(x_curr))

def estimate_x1(f, x0):
    sign = "+" if f(x0) < 0 else "-"
    while True:
        if sign == '+':
            if f(x0) > 0:
                return x0
        else:
            if f(x0) < 0:
                return x0
        x0 = x0 + 0.01

def secant_method(f, x0, eps):
    #x1 = newton_method(f, f_prime, x0)
    x1 = estimate_x1(f, x0)
    x_prev = x0
    x_curr = x1
    while True:
        if abs(x_curr-x_prev)  < eps:
            break
        x_prev, x_curr = x_curr, x_curr - (f(x_curr)*(x_curr-x_prev)/(f(x_curr)-f(x_prev)))
    print(f"The value of f in point {x_curr} is: {f(x_curr)}")
    return x_curr

x0 = 0
x1 = 0.5
print(secant_method(f, x0, 1e-3))