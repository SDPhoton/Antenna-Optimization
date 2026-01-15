import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
import time

# --- CONFIGURATION ---
# Physics/Geometry Settings
NUM_SEGMENTS = 12       # Number of wire segments
SEGMENT_LENGTH = 0.1    # Length of each segment (meters)
WAVELENGTH = 4.0        # Target wavelength (approx 75 MHz)
                        # We want a resonant structure roughly Lambda/4 total length
                        # but compressed into a small box.

# Genetic Algorithm Settings
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.2 # How much an angle changes (radians)
ELITISM_COUNT = 2       # Keep the best N solutions exactly as is

# --- THE ANTENNA CLASS ---

class Antenna:
    def __init__(self):
        # Genes: Array of (theta, phi) pairs for each segment
        # theta: angle from Z-axis (0 to pi)
        # phi: angle in XY plane (0 to 2pi)
        # We use relative angles to create smooth, organic shapes
        self.genes = np.random.uniform(-np.pi/2, np.pi/2, (NUM_SEGMENTS, 2))
        self.fitness = 0.0
        self.points = None # To cache 3D coordinates

    def compute_geometry(self):
        """Calculates the (x, y, z) coordinates of the wire segments."""
        points = [[0.0, 0.0, 0.0]] # Start at origin
        
        # Initial absolute orientation (pointing up)
        current_theta = 0.0 
        current_phi = 0.0
        
        # Accumulate coordinates
        x, y, z = 0.0, 0.0, 0.0
        
        for i in range(NUM_SEGMENTS):
            # Genes act as relative changes to angles (differential geometry)
            d_theta, d_phi = self.genes[i]
            
            # Update angles (accumulate)
            current_theta += d_theta
            current_phi += d_phi
            
            # Spherical to Cartesian conversion for the segment vector
            # dx = r * sin(theta) * cos(phi)
            # dy = r * sin(theta) * sin(phi)
            # dz = r * cos(theta)
            
            # Using absolute theta/phi for the vector direction
            dx = SEGMENT_LENGTH * np.sin(current_theta) * np.cos(current_phi)
            dy = SEGMENT_LENGTH * np.sin(current_theta) * np.sin(current_phi)
            dz = SEGMENT_LENGTH * np.cos(current_theta)
            
            x += dx
            y += dy
            z += dz
            
            points.append([x, y, z])
            
        self.points = np.array(points)
        return self.points

    def calculate_fitness(self):
        """
        Evaluates the antenna.
        Goal: 
        1. High Gain at Horizon (90 deg elevation).
        2. Low Gain at Zenith (0 deg elevation) - we want a donut pattern.
        3. Compactness: Penalty for being too tall (Z-height).
        """
        if self.points is None:
            self.compute_geometry()
            
        # 1. Physics Approximation (Array Factor)
        # Treat each segment midpoint as a point source
        midpoints = (self.points[:-1] + self.points[1:]) / 2.0
        
        # Assume a traveling wave current: Phase decreases with distance along wire
        # Distance of each midpoint along the wire
        dist_along_wire = np.arange(NUM_SEGMENTS) * SEGMENT_LENGTH + (SEGMENT_LENGTH/2)
        k = 2 * np.pi / WAVELENGTH
        current_phases = -k * dist_along_wire # Simple traveling wave model
        
        # Calculate Radiation at Horizon (Theta=90, Phi=0) - Target direction
        # r vector for observer at x-infinity
        # Phase shift due to geometry: k * (x_coord)
        geom_phases_horizon = k * midpoints[:, 0] 
        total_phase_horizon = current_phases + geom_phases_horizon
        # Vector sum of phasors
        E_horizon = np.abs(np.sum(np.exp(1j * total_phase_horizon)))
        
        # Calculate Radiation at Zenith (Theta=0) - Null direction
        # Phase shift due to geometry: k * (z_coord)
        geom_phases_zenith = k * midpoints[:, 2]
        total_phase_zenith = current_phases + geom_phases_zenith
        E_zenith = np.abs(np.sum(np.exp(1j * total_phase_zenith)))
        
        # 2. Structural Constraints
        # Max Z height
        max_z = np.max(self.points[:, 2])
        min_z = np.min(self.points[:, 2])
        height = max_z - min_z
        
        # --- FITNESS FUNCTION ---
        # Reward Horizon gain, Penalize Zenith gain
        rf_score = (E_horizon ** 2) - (E_zenith ** 2)
        
        # Constraint: We want it COMPACT. 
        # If we didn't add this, it would just make a straight wire.
        # We force it to fit in a box 1/2 the height of a natural monopole.
        height_penalty = 0
        target_height = (NUM_SEGMENTS * SEGMENT_LENGTH) * 0.4 
        if height > target_height:
            height_penalty = (height - target_height) * 10.0
            
        self.fitness = rf_score - height_penalty
        
        # Sanity floor
        if self.fitness < 0: self.fitness = 0.0
        
        return self.fitness

# --- GA OPERATORS ---

def crossover(parent1, parent2):
    """Single point crossover."""
    child = Antenna()
    cut = np.random.randint(1, NUM_SEGMENTS-1)
    child.genes = np.vstack((parent1.genes[:cut], parent2.genes[cut:]))
    return child

def mutate(antenna):
    """Gaussian mutation on genes."""
    if np.random.rand() < MUTATION_RATE:
        # Select a random segment to bend
        idx = np.random.randint(0, NUM_SEGMENTS)
        # Apply random change to theta and phi
        perturbation = np.random.normal(0, MUTATION_STRENGTH, 2)
        antenna.genes[idx] += perturbation
        
        # Reset cached geometry
        antenna.points = None
        antenna.fitness = 0.0

def tournament_selection(population, k=3):
    """Select best parent from random sample of k."""
    sample = random.sample(population, k)
    best = max(sample, key=lambda x: x.fitness)
    return best

# --- MAIN LOOP ---

def run_optimization():
    print(f"--- Starting Antenna Evolution ---")
    print(f"Population: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Segments: {NUM_SEGMENTS}, Target Wavelength: {WAVELENGTH}m")
    
    # Initialize
    population = [Antenna() for _ in range(POPULATION_SIZE)]
    for ind in population:
        ind.calculate_fitness()
        
    best_fitness_history = []
    avg_fitness_history = []
    
    global_best_ant = None
    global_best_fit = -float('inf')

    start_time = time.time()

    for gen in range(GENERATIONS):
        # Sort population by fitness (Descending)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Stats
        current_best = population[0]
        if current_best.fitness > global_best_fit:
            global_best_fit = current_best.fitness
            global_best_ant = copy.deepcopy(current_best)
            
        avg_fit = sum(p.fitness for p in population) / POPULATION_SIZE
        best_fitness_history.append(current_best.fitness)
        avg_fitness_history.append(avg_fit)
        
        if gen % 5 == 0 or gen == GENERATIONS - 1:
            print(f"Gen {gen}: Best Fitness = {current_best.fitness:.4f} | Avg = {avg_fit:.4f}")
            
        # Elitism
        new_population = [copy.deepcopy(p) for p in population[:ELITISM_COUNT]]
        
        # Breeding
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2)
            mutate(child)
            child.calculate_fitness() # Calc immediately for simplicity
            new_population.append(child)
            
        population = new_population

    end_time = time.time()
    print(f"\nOptimization Complete in {end_time - start_time:.2f} seconds.")
    
    return global_best_ant, best_fitness_history, avg_fitness_history

# --- VISUALIZATION ---

def plot_results(best_ant, best_hist, avg_hist):
    # 1. Plot Fitness History
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(best_hist, label="Best Fitness")
    plt.plot(avg_hist, label="Avg Fitness", linestyle='--')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.title("Evolutionary Progress")
    plt.legend()
    plt.grid(True)
    
    # 2. Plot 3D Antenna Structure
    points = best_ant.compute_geometry()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    ax = plt.subplot(1, 2, 2, projection='3d')
    
    # Plot wire
    ax.plot(x, y, z, color='b', linewidth=2, marker='o', markersize=3, label='Antenna Wire')
    
    # Plot Start Point
    ax.scatter(0, 0, 0, color='g', s=100, label='Feed Point')
    
    # Calculate bounds for equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Evolved Antenna Shape")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print simplified 'gene' output
    print("\nBest Antenna Segment Angles (Theta, Phi radians):")
    print(best_ant.genes)

if __name__ == "__main__":
    best_antenna, fit_hist, avg_hist = run_optimization()
    plot_results(best_antenna, fit_hist, avg_hist)
