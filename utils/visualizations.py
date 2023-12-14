import matplotlib.pyplot as plt


def plot_distribution_duplicate_ratio(dataframe, column_name) -> plt.Figure:
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

    plt.title(f"Distribution of Node Overlap Ratio")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.xlim(-0.02, 0.5)

    plt.grid(axis="y")

    # Return the Matplotlib figure object
    return plt.gcf()


def plot_distribution_unique_nodes(node_list, unique_node_counts):
    total_num_nodes = len(node_list)

    duplicate_values = list(unique_node_counts.values())
    ratio_unique_nodes = sum(duplicate_values) / total_num_nodes
    num_unique_nodes = total_num_nodes - sum(duplicate_values)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plt.hist(duplicate_values, bins=200)

    # Create a box to display statistics
    stats_box = f"Number of Nodes: {total_num_nodes:.4f}\nNumber of Unique Nodes: {num_unique_nodes:.4f}\nRatio of Duplicate Nodes: {ratio_unique_nodes:.4f}"
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

    plt.title("Frequency of Unique Nodes")
    plt.xlabel("Frequency")
    plt.ylabel("Frequency")
    plt.grid(axis="y")

    # Return the Matplotlib figure object
    return plt.gcf()


def plot_bars_unique_nodes(node_list, unique_node_counts):
    total_num_nodes = len(node_list)

    duplicate_values = list(unique_node_counts.values())
    ratio_unique_nodes = sum(duplicate_values) / total_num_nodes
    num_unique_nodes = total_num_nodes - sum(duplicate_values)
    num_unique_duplicates = len(duplicate_values)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plt.plot(range(len(duplicate_values)), sorted(duplicate_values))

    # Create a box to display statistics
    stats_box = f"Number of Nodes: {total_num_nodes}\n\
Number of Unique Nodes: {num_unique_nodes}\n\
Count of Duplicate Nodes: {num_unique_duplicates}\n\
Ratio of Duplicate Nodes: {ratio_unique_nodes:.4f}"
    plt.text(
        0.05,
        0.95,
        stats_box,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=12,
    )

    plt.title("Frequency of Duplicate Nodes")
    plt.xlabel("Node Id")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.yscale("log")

    # Return the Matplotlib figure object
    return plt.gcf()
