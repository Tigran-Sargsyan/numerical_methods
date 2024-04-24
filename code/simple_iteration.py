import numpy as np
import matplotlib.pyplot as plt

def simple_iteration(g, x0, eps):
    x = x0
    estimations = [x]
    while True:
        x1 = g(x)
        estimations.append(x1)
        #If x is too big return "The method doesn't converge"
        if abs(x1) > 1000:
            estimations = []
            return "The method doesn't converge", estimations
        if abs(x1 - x) < eps:
            return x1, estimations
        x = x1

def g(x):
    return x**2-2*x+2

# Improved plotting function that removes empty subplots
def plot_filtered_iterations(estimations):
    # Filter out the empty or non-converging estimations
    filtered_estimations = [est for est in estimations if est != "The method doesn't converge" and est]
    
    # Determine the number of rows and columns for subplots based on filtered list
    n_estimations = len(filtered_estimations)
    n_rows = int(np.ceil(np.sqrt(n_estimations)))
    n_cols = int(np.ceil(n_estimations / n_rows))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    # Flatten the axes array for easy iteration
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n_estimations:
            estimation = filtered_estimations[i]
            # Scatter plot for all estimations
            ax.scatter(estimation[:-1], estimation[1:], color='blue', s=20, label='Iterations')
            # Emphasize start and end points
            ax.scatter(estimation[0], estimation[1], color='red', s=100, label='Start', zorder=3)
            ax.scatter(estimation[-2], estimation[-1], color='green', s=100, label='End', zorder=3)
            
            # Plotting the functions
            x_vals = np.linspace(min(estimation) - 1, max(estimation) + 1, 400)
            ax.plot(x_vals, g(x_vals), label='g(x)', color='black', linewidth=2)
            ax.plot(x_vals, x_vals, label='y = x', linestyle='--', color='gray', linewidth=2)
            
            ax.set_title(f'Estimation starting at {estimation[0]:.3f}', fontsize=12)
            ax.legend()
        else:
            # Remove any excess axes
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

convergences = [simple_iteration(g, x0, 1e-5)[0] for x0 in [-0.3, 0.003, 0.5, 1.7, 2, 2.1]]
estimations = [simple_iteration(g, x0, 1e-5)[1] for x0 in [-0.3, 0.003, 0.5, 1.7, 2, 2.1]]

# Now let's call the function to plot filtered iterations
plot_filtered_iterations(estimations)
