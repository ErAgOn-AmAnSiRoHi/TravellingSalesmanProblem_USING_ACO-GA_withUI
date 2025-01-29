import sys
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modify script to take file path from command line argument
if len(sys.argv) < 2:
    raise ValueError("Please provide a CSV file path as an argument")
file_path = sys.argv[1]


def read_csv_to_floats(file_path):
    float_pairs = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            float_pairs.append([float(value) for value in row])
    return float_pairs

class SolveTSPUsingACO:
    # [Previous Edge and Ant class definitions remain the same]
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges, start_node):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.start_node = start_node
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [self.start_node]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, file_path, start_node=0, mode='ACS', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, 
                 alpha=1.0, beta=3.0, rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        
        # Load nodes from CSV
        df = pd.read_csv(file_path, header=None)
        self.nodes = df.values.tolist()
        self.num_nodes = len(self.nodes)
        self.start_node = start_node

        self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(
                    i, j, math.sqrt(
                        pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                    initial_pheromone
                )
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges, self.start_node) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _elitist(self):
        path = []
        cost = []
        for step in range(self.steps):
            self.tours = []
            for ant in self.ants:
                ant.find_tour()
                distance = ant.get_distance()
                self._add_pheromone(ant.tour, distance)
                l = [self.nodes[i] for i in ant.tour]
                self.tours.append(l)
                if distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = distance
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            path.append(self.global_best_tour)
            cost.append(self.global_best_distance)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
        self._animate_results(path, cost)

    def run(self):
        print('Started : {0}'.format(self.mode))
        self._elitist()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def _animate_results(self, paths, costs, animation_filename="abc"):
        xx = [node[0] for node in self.nodes]
        yy = [node[1] for node in self.nodes]
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, wspace=0.45, hspace=0.35)

        # Plot city points
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(xx, yy, c='r', edgecolors='black')
        ax1.set_xlabel('x (kms)')
        ax1.set_ylabel('y (kms)')
        ax1.set_title('Cities', fontweight='bold', pad=10)

        # Plot min cost per iteration
        ax2 = fig.add_subplot(gs[1, 0])
        costline, = ax2.plot([], [], 'b-')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Distance (kms)')
        ax2.set_ylim(min(costs) * 0.9, max(costs) * 1.1)
        ax2.set_xlim(0, len(costs) + 1)
        ax2.set_title('Best path distance per iteration', fontweight='bold', pad=10)

        # Plot best path
        ax3 = fig.add_subplot(gs[:, 1:])
        ax3.scatter(xx, yy, c='r', edgecolors='black')
        ax3.set_xlabel('x (kms)')
        ax3.set_ylabel('y (kms)')
        title = ax3.set_title('Best Path Cost', fontweight='bold', fontsize=13, pad=10)
        
        # Create an empty line for the path
        path_line, = ax3.plot([], [], '--', alpha=0.9)

        def animate(iter):
            # Update cost plot
            if iter < len(costs):
                costline.set_data(list(range(1, iter + 2)), costs[:iter + 1])
            
            # Update path plot
            if iter < len(costs):
                x = [xx[i] for i in paths[iter]]
                y = [yy[j] for j in paths[iter]]
                x.append(x[0])  # Close the loop
                y.append(y[0])
                path_line.set_data(x, y)
                title.set_text(f"Best Path Cost : {round(costs[iter], 2)} kms")
            elif iter == len(costs):
                x = [self.nodes[i][0] for i in self.global_best_tour]
                y = [self.nodes[i][1] for i in self.global_best_tour]
                x.append(x[0])
                y.append(y[0])
                path_line.set_data(x, y)
                path_line.set_color('#000')
                path_line.set_linestyle('-')
                title.set_text(f"Best Path Cost : {round(self.global_best_distance, 2)} kms")
            
            return costline, path_line, title

        ani = FuncAnimation(fig, animate, frames=len(costs) + 1, interval=100, blit=True, repeat=False)
        ani.save(f"static/gifs/{animation_filename}.gif", writer='pillow')

# Example usage
aco = SolveTSPUsingACO(file_path=file_path, start_node=0)
aco.run()