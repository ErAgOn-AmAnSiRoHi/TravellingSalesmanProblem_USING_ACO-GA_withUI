# Travelling Salesman Problem using ACO & GA with Interactive UI

## Overview
This project provides an **interactive UI** to solve the **Travelling Salesman Problem (TSP)** using **Ant Colony Optimization (ACO)** and **Genetic Algorithm (GA)**. In addition to solving TSP, the UI also offers **statistical insights** on the uploaded dataset, making it an intuitive tool for both algorithmic exploration and data analysis.

## Features
- **TSP Optimization**: Solve TSP using ACO and GA algorithms.
- **Interactive UI**: Easy-to-use interface to upload datasets and visualize results.
- **Preloaded Dataset for Instant Execution**: A small dataset (`small_tsp.csv`) is included by default so users can instantly run simulations without uploading their own data.
- **Statistical Insights**:
  - Central tendency and dispersion measures
  - Correlation analysis
  - Clustering (DBSCAN, K-Means)
  - Outlier detection
  - Convex hull visualization
  - Distance metrics analysis
  - Various statistical plots (scatter plot, histograms, density plots, etc.)

## Getting Started

### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install pandas numpy matplotlib flask tkinter scipy seaborn scikit-learn shapely networkx
```

### Running the Application
1. Clone the repository:
```bash
git clone https://github.com/yourusername/TravellingSalesmanProblem_USING_ACO-GA_withUI.git
cd TravellingSalesmanProblem_USING_ACO-GA_withUI
```
2. Run the application:
```bash
python app.py
```
3. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
4. You can either:
   - **Run the algorithms instantly** using the **preloaded dataset (`small_tsp.csv`)** to get a quick overview.
   - **Upload your own dataset** in `.csv` or `.tsp` format to see how the algorithms perform on your data.

## Directory Structure
```
TravellingSalesmanProblem_USING_ACO-GA_withUI
├── aco.py                      # Ant Colony Optimization implementation
├── app.py                      # Flask application entry point
├── ga.py                       # Genetic Algorithm implementation
├── static
│   ├── css
│   │   └── styles.css          # UI styling
│   ├── gifs
│   │   ├── abc.gif
│   │   └── xyz.gif             # Loading and result animations
│   └── inferences              # Precomputed statistical inferences
│       ├── central_tendency.txt
│       ├── convex_hull.png
│       ├── convex_hull.txt
│       ├── correlation_analysis.txt
│       ├── data_info.txt
│       ├── dbscan_clustering.png
│       ├── density_plot.png
│       ├── dispersion_measures.txt
│       ├── distance_metrics.txt
│       ├── histograms.png
│       ├── kmeans_clustering.png
│       ├── morans_i.txt
│       ├── nearest_neighbor_distances.png
│       ├── nearest_neighbor_stats.txt
│       ├── outlier_detection.png
│       ├── outliers.txt
│       └── scatter_plot.png
├── statistical_inferences.py    # Statistical analysis module
├── templates                   # HTML templates for UI
│   ├── index.html
│   ├── loading.html
│   ├── result.html
│   └── stats.html
└── uploads                     # Uploaded and sample datasets (we have provided some small as well as medium-sized datasets for your ease of use.)
    ├── att48.tsp
    ├── dj38.tsp
    └── small_tsp.csv           # Default dataset for instant execution
```

## Contribution
Feel free to fork this repository, make improvements, and submit pull requests!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **ACO & GA Algorithms**: Inspired by classical optimization techniques.
- **Flask Framework**: Used for the UI implementation.
- **Matplotlib & Seaborn**: Used for data visualization.

