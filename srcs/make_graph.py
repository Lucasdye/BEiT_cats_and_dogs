from matplotlib import pyplot as plt

def make_graph(x: list[int], y: list[int | float], label_x: str, label_y: str, ax: plt.Axes):
    # Plot the data on the provided axes and set labels
    ax.plot(x, y)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.scatter(x, y, color="red", zorder=5)


 