import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import distance_matrix
from shapely.geometry import MultiPoint
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import warnings
import networkx as nx
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
def ensure_output_directory():
    output_dir = "static/inferences"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def write_to_file(filename, content):
    output_dir = ensure_output_directory()
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(content)

def load_data(filepath):
    df = pd.read_csv(filepath, names=['x', 'y'])
    output = "Dataset Information:\n"
    output += f"First 5 rows of the dataset:\n{df.head().to_string()}\n\n"
    output += f"Dataset Info:\n{df.info(buf=None, max_cols=None, memory_usage=None, show_counts=None)}\n"
    write_to_file('data_info.txt', output)
    return df

def central_tendency(df):
    mean_x, mean_y = df['x'].mean(), df['y'].mean()
    median_x, median_y = df['x'].median(), df['y'].median()
    mode_x, mode_y = df['x'].mode().values, df['y'].mode().values
    
    output = "Central Tendency Measures:\n"
    output += f"Mean (Centroid) - x: {mean_x}, y: {mean_y}\n"
    output += f"Median - x: {median_x}, y: {median_y}\n"
    output += f"Mode - x: {mode_x}, y: {mode_y}\n"
    write_to_file('central_tendency.txt', output)

def dispersion_measures(df):
    range_x, range_y = df['x'].max() - df['x'].min(), df['y'].max() - df['y'].min()
    variance_x, variance_y = df['x'].var(), df['y'].var()
    std_dev_x, std_dev_y = df['x'].std(), df['y'].std()
    iqr_x, iqr_y = stats.iqr(df['x']), stats.iqr(df['y'])
    
    output = "Dispersion Measures:\n"
    output += f"Range - x: {range_x}, y: {range_y}\n"
    output += f"Variance - x: {variance_x}, y: {variance_y}\n"
    output += f"Standard Deviation - x: {std_dev_x}, y: {std_dev_y}\n"
    output += f"IQR - x: {iqr_x}, y: {iqr_y}\n"
    write_to_file('dispersion_measures.txt', output)

def correlation_analysis(df):
    pearson_corr, p_value = stats.pearsonr(df['x'], df['y'])
    output = "Correlation Analysis:\n"
    output += f"Pearson Correlation between x and y: {pearson_corr} (p-value: {p_value})\n"
    write_to_file('correlation_analysis.txt', output)

def plot_distributions(df):
    output_dir = ensure_output_directory()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['x'], kde=True, bins=30, color='skyblue')
    plt.title('Histogram of X Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['y'], kde=True, bins=30, color='salmon')
    plt.title('Histogram of Y Coordinates')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'histograms.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', data=df, s=50, color='purple')
    plt.title('Scatter Plot of Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
    plt.close()

def detect_outliers(df, threshold=3):
    output_dir = ensure_output_directory()
    
    df['z_score_x'] = np.abs(stats.zscore(df['x']))
    df['z_score_y'] = np.abs(stats.zscore(df['y']))
    outliers_x = df[df['z_score_x'] > threshold]
    outliers_y = df[df['z_score_y'] > threshold]
    
    output = "Outlier Detection:\n"
    output += f"Number of outliers in x: {outliers_x.shape[0]}\n"
    output += f"Number of outliers in y: {outliers_y.shape[0]}\n"
    write_to_file('outliers.txt', output)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', data=df, label='Normal Points', color='blue')
    sns.scatterplot(x='x', y='y', data=outliers_x, label='Outliers in X', color='red')
    sns.scatterplot(x='x', y='y', data=outliers_y, label='Outliers in Y', color='green')
    plt.title('Outlier Detection')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'outlier_detection.png'))
    plt.close()

def spatial_statistics(df):
    output_dir = ensure_output_directory()
    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=df['x'], y=df['y'], cmap="Reds", shade=True, bw_adjust=0.5)
    plt.title('Density Plot of Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(os.path.join(output_dir, 'density_plot.png'))
    plt.close()

def clustering(df):
    output_dir = ensure_output_directory()
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['x', 'y']])
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='Set1', s=50)
    plt.title('K-Means Clustering')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_dir, 'kmeans_clustering.png'))
    plt.close()

    dbscan = DBSCAN(eps=5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(df[['x', 'y']])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', hue='dbscan_cluster', data=df, palette='Set2', s=50)
    plt.title('DBSCAN Clustering')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_dir, 'dbscan_clustering.png'))
    plt.close()

def morans_I(df, distance_threshold=10):
    n = len(df)
    tree = KDTree(df[['x', 'y']].values)
    neighbors = tree.query_ball_tree(tree, r=distance_threshold)
    W = np.zeros((n, n))
    for i, nbr in enumerate(neighbors):
        for j in nbr:
            if i != j:
                W[i][j] = 1
    W_sum = W.sum()
    
    x_mean, y_mean = df['x'].mean(), df['y'].mean()
    num_x, denom_x = 0, sum((df['x'] - x_mean) ** 2)
    num_y, denom_y = 0, sum((df['y'] - y_mean) ** 2)
    for i in range(n):
        for j in range(n):
            if W[i][j] != 0:
                num_x += W[i][j] * (df['x'][i] - x_mean) * (df['x'][j] - x_mean)
                num_y += W[i][j] * (df['y'][i] - y_mean) * (df['y'][j] - y_mean)
    I_x = (n / W_sum) * (num_x / denom_x)
    I_y = (n / W_sum) * (num_y / denom_y)
    
    output = "Moran's I Analysis:\n"
    output += f"Moran's I for X: {I_x}\n"
    output += f"Moran's I for Y: {I_y}\n"
    write_to_file('morans_i.txt', output)

def distance_metrics(df):
    pairwise_dist = distance_matrix(df[['x', 'y']], df[['x', 'y']])
    min_dist = np.min(pairwise_dist + np.eye(len(df)) * np.max(pairwise_dist))
    max_dist = np.max(pairwise_dist)
    
    output = "Distance Metrics:\n"
    output += f"Minimum Pairwise Distance: {min_dist}\n"
    output += f"Maximum Pairwise Distance: {max_dist}\n"
    write_to_file('distance_metrics.txt', output)

def convex_hull(df):
    output_dir = ensure_output_directory()
    
    points = MultiPoint(df[['x', 'y']].values)
    hull = points.convex_hull
    hull_perimeter, hull_area = hull.length, hull.area
    
    output = "Convex Hull Analysis:\n"
    output += f"Convex Hull Perimeter: {hull_perimeter}\n"
    output += f"Convex Hull Area: {hull_area}\n"
    write_to_file('convex_hull.txt', output)
    
    x_hull, y_hull = hull.exterior.coords.xy
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], s=50, color='blue', label='Points')
    plt.plot(x_hull, y_hull, color='red', label='Convex Hull')
    plt.title('Convex Hull of the Dataset')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convex_hull.png'))
    plt.close()

def nearest_neighbor_stats(df):
    output_dir = ensure_output_directory()
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(df[['x', 'y']].values)
    distances, _ = nbrs.kneighbors(df[['x', 'y']].values)
    nearest_distances = distances[:, 1]
    average_nearest_distance = nearest_distances.mean()
    
    output = "Nearest Neighbor Statistics:\n"
    output += f"Average Nearest Neighbor Distance: {average_nearest_distance}\n"
    write_to_file('nearest_neighbor_stats.txt', output)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(nearest_distances, kde=True, bins=30, color='purple')
    plt.title('Distribution of Nearest Neighbor Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'nearest_neighbor_distances.png'))
    plt.close()

def analyze(filepath):
    # Create output directory
    ensure_output_directory()
    
    # Run all analyses
    df = load_data(filepath)
    central_tendency(df)
    dispersion_measures(df)
    correlation_analysis(df)
    plot_distributions(df)
    detect_outliers(df)
    spatial_statistics(df)
    clustering(df)
    morans_I(df)
    distance_metrics(df)
    convex_hull(df)
    nearest_neighbor_stats(df)

# Modify only the main execution part at the bottom of the file:
if __name__ == "__main__":
    # Check if filepath is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python statistical_inferences.py <filepath>")
        sys.exit(1)
    
    # Get filepath from command line argument
    filepath = sys.argv[1]
    
    # Verify file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
    
    # Run analysis
    analyze(filepath)