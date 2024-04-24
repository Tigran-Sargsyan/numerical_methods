import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, lambdify

def newton_raphson(f, x0, eps):
    x = symbols('x')
    f_prime = diff(f, x)
    f_num = lambdify(x, f)
    f_prime_num = lambdify(x, f_prime)
    
    xn = x0
    estimations = [xn]
    xn1 = xn - f_num(xn) / (f_prime_num(xn) if f_prime_num(xn) != 0 else float('inf'))
    
    while abs(xn1 - xn) >= eps:
        xn = xn1
        # Check if the derivative is zero
        f_prime_val = f_prime_num(xn)
        if f_prime_val == 0:
            print(f"Derivative is zero at x = {xn}. The method failed to converge.")
            return None  # Return None or consider handling this case differently
        if abs(f_prime_val) > 1000:
            print(f"Derivative is too big at x = {xn}. The method failed to converge.")
            return None
        estimations.append(xn)
        xn1 = xn - f_num(xn) / f_prime_val
    
    return xn1, estimations

def plot_filtered_iterations(estimation_tuples):
    # Filter out tuples where the method did not converge, and extract just the estimations for plotting
    filtered_estimations = [est[1] for est in estimation_tuples if est[0] is not None and len(est[1]) > 1]

    n_estimations = len(filtered_estimations)
    if n_estimations == 0:
        print("No valid estimations to plot.")
        return

    n_rows = int(np.ceil(np.sqrt(n_estimations)))
    n_cols = int(np.ceil(n_estimations / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()

    for i, estimation in enumerate(filtered_estimations):
        if estimation:
            axes[i].scatter(estimation[:-1], estimation[1:], color='blue', s=20, label='Iterations', zorder=2)
            axes[i].scatter(estimation[0], estimation[1], color='red', s=100, label='Start', zorder=3)
            if len(estimation) > 2:  # Ensure there's an "end" point to plot
                axes[i].scatter(estimation[-2], estimation[-1], color='green', s=100, label='End', zorder=3)
            x_vals = np.linspace(min(estimation + [2]) - 3, max(estimation + [2]) + 1, 400)
            g_lambdified = lambdify(symbols('x'), f, 'numpy')
            axes[i].plot(x_vals, g_lambdified(x_vals), label='f(x)', color='black', linewidth=2)
            axes[i].plot(x_vals, [0]*len(x_vals), label='y = x', linestyle='--', color='gray', linewidth=2)
            axes[i].set_title(f'Estimation starting at {estimation[0]:.3f}', fontsize=12)
            axes[i].legend()
        if i >= n_estimations:
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


x = symbols('x')
f = x**2 - 2*x - 2

convergences = [newton_raphson(f, x0, 1e-5)[0] for x0 in [-0.3, 0.003, 0.5, 1.7, 2, 2.1]]
estimations = [newton_raphson(f, x0, 1e-5) for x0 in [-0.3, 0.003, 0.5, 1.7, 2, 2.1]]

plot_filtered_iterations(estimations)