#PYTHON 
import numpy as np

class GA_C:
    def __init__(self):
        pass

    def initialize_random_solution(self, N, D, dayoff, A, B):
        X = np.zeros((N+1, D+1), dtype=np.int8)
        
        # Faster initialization with vectorized operations where possible
        for i in range(1, N+1):
            # Set random values for non-dayoff cells
            mask = (dayoff[i, 1:D+1] == 0)
            X[i, 1:D+1][mask] = np.random.randint(0, 5, size=np.sum(mask))
            
        # Fix consecutive night shift violations
        for i in range(1, N+1):
            for d in range(2, D+1):
                if X[i, d-1] == 4:
                    X[i, d] = 0
        
        # Optimize shift distribution
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                
                # Optimize using pre-calculated arrays for speed
                if count < A:
                    available_indices = [i for i in range(1, N+1) 
                                      if X[i, d] == 0 and dayoff[i, d] == 0 
                                      and (d == 1 or X[i, d-1] != 4)]
                    
                    # Use bulk assignment when possible
                    to_assign = min(A - count, len(available_indices))
                    if to_assign > 0:
                        selected = np.random.choice(available_indices, to_assign, replace=False)
                        X[selected, d] = shift
                
                elif count > B:
                    assigned = np.where((X[:, d] == shift) & (np.arange(N+1) > 0))[0]
                    to_remove = min(count - B, len(assigned))
                    if to_remove > 0:
                        selected = np.random.choice(assigned, to_remove, replace=False)
                        X[selected, d] = 0
        
        return X
    
    def evaluate_fitness(self, X, N, D, A, B, dayoff):
        # Use vectorized operations for faster evaluation
        penalty = 0
        
        # Dayoff violations (vectorized)
        dayoff_violations = np.sum((dayoff[1:N+1, 1:D+1] == 1) & (X[1:N+1, 1:D+1] != 0))
        penalty -= 1000 * dayoff_violations
        
        # Shift count constraints
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < A:
                    penalty -= 100 * (A - count)
                elif count > B:
                    penalty -= 100 * (count - B)
        
        # Night shift followed by work violations
        night_followed_violations = 0
        for i in range(1, N+1):
            for d in range(1, D):
                if X[i, d] == 4 and X[i, d+1] != 0:
                    night_followed_violations += 1
        penalty -= 1000 * night_followed_violations
        
        # Calculate max night shifts (vectorized)
        night_shifts_per_person = np.sum(X[1:N+1, 1:D+1] == 4, axis=1)
        max_night_shift = np.max(night_shifts_per_person) if night_shifts_per_person.size > 0 else 0
        
        return -max_night_shift + penalty
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_index]
    
    def crossover(self, parent1, parent2, crossover_rate=0.8):  # Increased crossover rate
        N, D = parent1.shape
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Row-wise crossover (faster than element-wise)
        for i in range(1, N):
            if np.random.rand() < crossover_rate:
                child1[i], child2[i] = parent2[i], parent1[i]
                
        return child1, child2
    
    def mutate(self, X, N, D, dayoff, A, B, mutation_rate=0.05):  # Reduced mutation rate
        X_new = X.copy()
        
        # Mutation phase - use vectorized operations where possible
        mutation_mask = np.random.rand(N+1, D+1) < mutation_rate
        mutation_mask[0, :] = False  # Don't mutate row 0
        mutation_mask[:, 0] = False  # Don't mutate column 0
        
        # Apply mutations considering constraints
        for i in range(1, N+1):
            for d in range(1, D+1):
                if mutation_mask[i, d]:
                    if dayoff[i, d] == 1:
                        X_new[i, d] = 0
                    elif d > 1 and X_new[i, d-1] == 4:
                        X_new[i, d] = 0
                    else:
                        X_new[i, d] = np.random.randint(0, 5)
        
        # Fix night shift violations
        for i in range(1, N+1):
            for d in range(1, D):
                if X_new[i, d] == 4:
                    X_new[i, d+1] = 0
        
        # Optimize shift distribution using bulk operations
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X_new[:, d] == shift)
                
                if count < A:
                    # Find eligible staff for assignment
                    eligible = np.zeros(N+1, dtype=bool)
                    for i in range(1, N+1):
                        if (X_new[i, d] == 0 and dayoff[i, d] == 0 and 
                            (d == 1 or X_new[i, d-1] != 4)):
                            eligible[i] = True
                    
                    eligible_indices = np.where(eligible)[0]
                    to_assign = min(A - count, len(eligible_indices))
                    
                    if to_assign > 0:
                        selected = np.random.choice(eligible_indices, to_assign, replace=False)
                        X_new[selected, d] = shift
                        
                        # Fix next day for night shifts
                        if shift == 4 and d < D:
                            X_new[selected, d+1] = 0
                
                elif count > B:
                    assigned = np.where((X_new[:, d] == shift) & (np.arange(N+1) > 0))[0]
                    to_remove = min(count - B, len(assigned))
                    
                    if to_remove > 0:
                        selected = np.random.choice(assigned, to_remove, replace=False)
                        X_new[selected, d] = 0
        
        return X_new

    def solve(self, N, D, A, B, dayoff):
        # Optimized parameters
        population_size = 30  # Reduced from 50
        generations = 15      # Reduced from 20
        mutation_rate = 0.05  # Reduced from 0.1
        
        # Use dtype=np.int8 to save memory
        population = [self.initialize_random_solution(N, D, dayoff, A, B) for _ in range(population_size)]
        
        best_fitness = float('-inf')
        best_solution = None
        
        for generation in range(generations):
            # Calculate fitness in parallel if possible
            fitness_scores = np.array([self.evaluate_fitness(ind, N, D, A, B, dayoff) for ind in population])
            
            # Track best solution
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_solution = population[current_best_idx].copy()
                
            # Elitism - keep the best solution
            new_population = [population[current_best_idx]]
            
            # Create new population
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, N, D, dayoff, A, B, mutation_rate)
                
                new_population.append(child1)
                
                if len(new_population) < population_size:
                    child2 = self.mutate(child2, N, D, dayoff, A, B, mutation_rate)
                    new_population.append(child2)
            
            population = new_population[:population_size]
        
        # Output solution in required format
        for i in range(1, N + 1):
            print(' '.join(str(int(best_solution[i, d])) for d in range(1, D + 1)))
        
        return best_solution

def main():
    # Efficient input parsing
    N, D, A, B = map(int, input().split())
    dayoff = np.zeros((N+1, D+1), dtype=np.int8)
    
    for i in range(1, N+1):
        days = list(map(int, input().split()))
        for day in days:
            if day == -1:
                break
            if 1 <= day <= D:
                dayoff[i, day] = 1
    
    # Solve problem
    ga = GA_C()
    ga.solve(N, D, A, B, dayoff)

if __name__ == "__main__":
    main()
