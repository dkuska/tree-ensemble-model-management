import matplotlib.pyplot as plt


def plot_distribution(dataframe, column_name) -> plt.Figure:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    dataframe[column_name].plot(
        kind="hist",
        bins=20,
        color="skyblue",
        edgecolor="black",
        ax=ax,
    )

    # Create a box to display statistics
    stats_box = f"Mean: {dataframe[column_name].mean():.4f}\nStd Dev: {dataframe[column_name].std():.4f}"
    plt.text(
        0.95,
        0.95,
        stats_box,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=12,
    )

    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.xlim(-0.02, 0.5)

    plt.grid(axis="y")

    # Return the Matplotlib figure object
    return plt.gcf()
