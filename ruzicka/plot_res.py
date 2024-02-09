import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()


def plot_heatmap(filename):
    df = pd.read_csv(filename)
    df = df.set_index(df.label)
    df = df.drop('label', axis=1)

    # Draw a heatmap with the numeric values in each cell
    # Plotting the heatmap
    clustermap = sns.clustermap(df,
                                pivot_kws=None, metric='cosine',  # method='centroid',
                                z_score=None, standard_scale=None,
                                figsize=(10, 10),  # cbar_kws={'ticks': []},
                                row_cluster=True, col_cluster=True,
                                row_linkage=None, col_linkage=None,
                                row_colors=None, col_colors=None,
                                mask=None, dendrogram_ratio=0.2,
                                # fmt='d',cmap = "Blues"
                                colors_ratio=0.03, tree_kws=None)
    clustermap.ax_heatmap.set_yticklabels(
        clustermap.ax_heatmap.get_yticklabels(), fontsize=7)
    # clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(False), fontsize=12)  # rotation=45, ha='right',
    # clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(False), fontsize=7)

    plt.title('Heatmap of the DataFrame')
    plt.savefig(f"{filename}.png")
    plt.show()
