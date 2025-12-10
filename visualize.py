# src/visualize.py
"""
Visualization helpers - saves plots to disk.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set(style='whitegrid')

def save_elbow(k_list, wcss_list, path):
    plt.figure(figsize=(7,5))
    plt.plot(k_list, wcss_list, '-o')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('WCSS (inertia)')
    plt.title('Elbow curve')
    plt.xticks(k_list)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_cluster_scatter(df, x_col, y_col, labels, centers, path):
    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab10', len(np.unique(labels)))
    sns.scatterplot(x=df[x_col], y=df[y_col], hue=labels, palette=palette, legend='brief', s=60)
    if centers is not None:
        # centers expected on scaled space; if plotting raw, centers should be mapped accordingly
        plt.scatter(centers[:,0], centers[:,1], c='black', s=200, marker='X', label='centroids')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Clusters: {} vs {}'.format(x_col, y_col))
    plt.legend(title='cluster')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_actual_vs_pred(y_true, y_pred, path, title='Actual vs Predicted'):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_residuals(y_true, y_pred, path):
    residuals = y_true - y_pred
    plt.figure(figsize=(7,4))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.savefig(path, bbox_inches='tight')
    plt.close()