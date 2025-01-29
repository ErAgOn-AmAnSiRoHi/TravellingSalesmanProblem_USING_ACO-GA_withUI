import sys
from tkinter import *
from tkinter.ttk import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import csv
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgo:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.CITY_COORD = np.array(self.read_csv_to_floats(csv_file))
        self.N = len(self.CITY_COORD)  # Number of cities based on the CSV data
        self.CITY_LABELS = list(range(self.N))
        self.CITY_DICT = {label: coord for (label, coord) in zip(self.CITY_LABELS, self.CITY_COORD)}

    def read_csv_to_floats(self, file_path):
        float_pairs = []
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                float_pairs.append([float(value) for value in row])
        return float_pairs

    def init(self, pop_size):
        def random_permutation():
            population = [list(np.random.permutation(self.CITY_LABELS)) for _ in range(pop_size)]
            return population
        return random_permutation()

    def fit(self, population):
        fitness = []
        for individual in population:
            distance = 0
            for i, city in enumerate(individual):
                s = self.CITY_DICT[individual[i-1]]
                t = self.CITY_DICT[individual[i]]
                distance += np.linalg.norm(s - t)
            fitness.append(1 / distance)
        return fitness

    def selection(self, population, fitness, n):
        def roulette():
            idx = np.arange(0, len(population))
            probabilities = fitness / np.sum(fitness)
            parents_idx = np.random.choice(idx, size=n, p=probabilities)
            parents = np.take(population, parents_idx, axis=0)
            return [(parents[i], parents[i + 1]) for i in range(0, len(parents) - 1, 2)]
        return roulette()

    def crossover(self, parents, crossover_rate=0.9):
        def ordered():
            children = []
            for pair in parents:
                if np.random.random() < crossover_rate:
                    for (parent1, parent2) in [(pair[0], pair[1]), (pair[1], pair[0])]:
                        points = np.random.randint(0, len(parent1), 2)
                        start, end = min(points), max(points)
                        segment1 = [x for x in parent1[start:end]]
                        segment2 = [x for x in parent2[end:] if x not in segment1]
                        segment3 = [x for x in parent2[:end] if x not in segment1]
                        child = segment3 + segment1 + segment2
                        children.append(child)
                else:
                    children.extend(pair)
            return children
        return ordered()

    def mutation(self, children, mutation_rate=0.05):
        def swap():
            for i, child in enumerate(children):
                if np.random.random() < mutation_rate:
                    a, b = np.random.randint(0, len(child), 2)
                    child[a], child[b] = child[b], child[a]
            return children
        return swap()

    def elitism(self, population, fitness, n):
        return [e[0] for e in sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:n]]

    def base_algorithm(self, pop_size, max_generations, elite_size=0, start_point=None):
        if start_point:
            population = [list(np.roll(self.CITY_LABELS, self.CITY_LABELS.index(start_point)))]
            population.extend(self.init(pop_size - 1))
        else:
            population = self.init(pop_size)
        
        # Store all generations in a list
        all_generations = []
        current_fitness = self.fit(population)
        all_generations.append((0, population, current_fitness))
        
        for g in range(max_generations):
            fitness = self.fit(population)
            elite = self.elitism(population, fitness, elite_size)
            parents = self.selection(population, fitness, pop_size - elite_size)
            children = self.crossover(parents)
            children = self.mutation(children)
            population = elite + children
            current_fitness = self.fit(population)
            all_generations.append((g + 1, population, current_fitness))
        
        return all_generations

    def ga(self, y_max, y_min, x_max, x_min, start_point):
        # Get all generations first
        generations = self.base_algorithm(pop_size=100, max_generations=500, elite_size=10, start_point=start_point)
        
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, wspace=0.45, hspace=0.35)

        ax3 = fig.add_subplot(gs[0, 0])
        ax3.set_xlabel('x (kms)')
        ax3.set_ylabel('y (kms)')
        ax3.set_title('Cities', fontweight='bold', pad=10)
        ax3.set_xlim([x_min, x_max])
        ax3.set_ylim([y_min, y_max])
        ax3.scatter(self.CITY_COORD[:, 0], self.CITY_COORD[:, 1], c='r', edgecolors='black', alpha=0.85)
        for i, coord in enumerate(self.CITY_COORD):
            ax3.annotate(i, (coord[0], coord[1]), textcoords="offset points", xytext=(0, 10), ha='center')

        x = []
        y_min_values = []
        y_mean = []

        ax0 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[:, 1:])

        def animate(frame_idx):
            ax0.clear()
            ax1.clear()
            ax0.set_title('Best path distance in every generation', fontweight='bold', pad=10)
            ax0.set_xlabel('Generations')
            ax0.set_ylabel('Distance (kms)')
            
            g, population, fitness = generations[frame_idx]
            x.append(g)
            dist = [1 / f for f in fitness]
            y_min_values.append(np.min(dist))
            y_mean.append(np.mean(dist))
            ax0.plot(x, y_min_values, color='blue', alpha=0.7, label='Fittest individual')
            ax0.legend(loc='upper right')
            ax1.set_title(f"Best Path Cost: {np.min(dist):.2f} kms")
            ax1.set_xlabel('x (kms)')
            ax1.set_ylabel('y (kms)')
            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])
            ax1.scatter(self.CITY_COORD[:, 0], self.CITY_COORD[:, 1], c='r', edgecolors='black', alpha=0.85)
            solution = max(zip(population, fitness), key=lambda x: x[1])[0]
            P = np.array([self.CITY_DICT[s] for s in solution] + [self.CITY_DICT[solution[0]]])
            ax1.plot(P[:, 0], P[:, 1], '--', c='black', alpha=0.85)
            return ax0, ax1

        anim = FuncAnimation(fig, animate, frames=len(generations), interval=50, repeat=False)
        anim.save('static/gifs/xyz.gif', writer='pillow')


    def run(self, start_point=None):
        x_min, x_max = self.CITY_COORD[:, 0].min() - 0.5, self.CITY_COORD[:, 0].max() + 0.5
        y_min, y_max = self.CITY_COORD[:, 1].min() - 0.5, self.CITY_COORD[:, 1].max() + 0.5
        self.ga(y_max, y_min, x_max, x_min, start_point)

# Check for CSV file input argument
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ga.py <path_to_csv>")
    else:
        csv_file = sys.argv[1]
        ga = GeneticAlgo(csv_file)
        ga.run(start_point=5)
