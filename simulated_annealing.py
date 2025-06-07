import numpy as np
import time
import math
import random
import sys

class StaffRosteringSolver:
    def __init__(self, N, D, A, B, days_off):
        self.N = N  # Number of staff
        self.D = D  # Number of days
        self.A = A  # Min staff per shift
        self.B = B  # Max staff per shift
        self.days_off = days_off  # Days off for each staff
        
        # Calculate lower bound for max night shifts
        total_night_shifts_needed = self.D * self.A
        self.theoretical_min = math.ceil(total_night_shifts_needed / self.N)
        
        # Track best solution
        self.best_solution = None
        self.best_max_nights = float('inf')
        
    def initialize_solution(self):
        """Create an initial feasible solution"""
        # Create empty solution matrix
        X = np.zeros((self.N + 1, self.D + 1), dtype=int)
        
        # First, mark all days off
        for i in range(1, self.N + 1):
            for d in self.days_off[i]:
                X[i, d] = 0  # Mark as day off
        
        # Initialize shift counts for each day
        shift_counts = np.zeros((self.D + 1, 5), dtype=int)  # [day][shift]
        
        # Track night shifts for each staff
        staff_night_shifts = np.zeros(self.N + 1, dtype=int)
        
        # Track if staff worked night shift the previous day
        worked_night_previous = np.zeros(self.N + 1, dtype=bool)
        
        # Initialize solution intelligently
        # First handle night shifts
        for d in range(1, self.D + 1):
            # Sort staff by number of night shifts (ascending)
            available_staff = [
                i for i in range(1, self.N + 1) 
                if d not in self.days_off[i] and not worked_night_previous[i]
            ]
            
            available_staff.sort(key=lambda i: staff_night_shifts[i])
            
            # Assign night shifts (shift 4) to meet minimum requirement
            for i in range(min(len(available_staff), self.A)):
                staff_id = available_staff[i]
                X[staff_id, d] = 4
                shift_counts[d, 4] += 1
                staff_night_shifts[staff_id] += 1
                
                # Mark next day as off due to night shift
                if d < self.D:
                    worked_night_previous[staff_id] = True
            
            # Reset night shift tracking for next day
            if d < self.D:
                worked_night_previous = np.zeros(self.N + 1, dtype=bool)
                for i in range(1, self.N + 1):
                    if X[i, d] == 4:
                        worked_night_previous[i] = True
        
        # Now assign remaining shifts 1-3
        for d in range(1, self.D + 1):
            for shift in range(1, 4):  # Morning, afternoon, evening
                # Find available staff for this shift
                available_staff = [
                    i for i in range(1, self.N + 1) 
                    if d not in self.days_off[i] and 
                    X[i, d] == 0 and  # Not already assigned
                    (d == 1 or X[i, d-1] != 4)  # Not worked night shift yesterday
                ]
                
                # Shuffle to randomize assignments
                random.shuffle(available_staff)
                
                # Assign staff to meet minimum requirement
                needed = max(0, self.A - shift_counts[d, shift])
                for i in range(min(len(available_staff), needed)):
                    staff_id = available_staff[i]
                    X[staff_id, d] = shift
                    shift_counts[d, shift] += 1
        
        # Perform repair to ensure all constraints are satisfied
        X = self.repair_solution(X)
        return X
    
    def repair_solution(self, X):
        """Fix any constraint violations"""
        X = np.copy(X)
        
        # Fix day off violations
        for i in range(1, self.N + 1):
            for d in self.days_off[i]:
                X[i, d] = 0
        
        # Fix night shift + day off violations
        for i in range(1, self.N + 1):
            for d in range(1, self.D):
                if X[i, d] == 4:
                    X[i, d+1] = 0
        
        # Fix shift counts
        for d in range(1, self.D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                
                # If too few staff, add more
                while count < self.A:
                    # Find available staff
                    available = [
                        i for i in range(1, self.N + 1) 
                        if X[i, d] == 0 and  # Not assigned
                        d not in self.days_off[i] and  # Not day off
                        (d == 1 or X[i, d-1] != 4)  # Not after night shift
                    ]
                    
                    if not available:
                        print(f"Warning: Cannot satisfy min staff requirement for day {d}, shift {shift}", file=sys.stderr)
                        break
                    
                    # Choose staff with fewer total shifts
                    staff_shifts = {i: np.sum(X[i] > 0) for i in available}
                    i = min(staff_shifts, key=staff_shifts.get)
                    
                    X[i, d] = shift
                    if shift == 4 and d < self.D:
                        X[i, d+1] = 0
                    count += 1
                
                # If too many staff, remove some
                while count > self.B:
                    # Find assigned staff
                    assigned = [i for i in range(1, self.N + 1) if X[i, d] == shift]
                    
                    if not assigned:
                        break
                    
                    # Choose staff with more total shifts
                    staff_shifts = {i: np.sum(X[i] > 0) for i in assigned}
                    i = max(staff_shifts, key=staff_shifts.get)
                    
                    X[i, d] = 0
                    count -= 1
        
        return X
    
    def evaluate_solution(self, X):
        """Calculate max night shifts and check constraints"""
        # Count night shifts per staff
        night_shifts = [np.sum(X[i] == 4) for i in range(1, self.N + 1)]
        max_nights = max(night_shifts) if night_shifts else 0
        
        # Check constraints
        violations = 0
        
        # Check day off violations
        for i in range(1, self.N + 1):
            for d in self.days_off[i]:
                if X[i, d] != 0:
                    violations += 100
        
        # Check night shift + day off violations
        for i in range(1, self.N + 1):
            for d in range(1, self.D):
                if X[i, d] == 4 and X[i, d+1] != 0:
                    violations += 100
        
        # Check shift counts
        for d in range(1, self.D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < self.A:
                    violations += 10 * (self.A - count)
                elif count > self.B:
                    violations += 10 * (count - self.B)
        
        # Secondary objective: balance night shifts
        if night_shifts:
            avg_nights = sum(night_shifts) / len(night_shifts)
            variance = sum((n - avg_nights) ** 2 for n in night_shifts)
            balance_penalty = variance / 10
        else:
            balance_penalty = 0
        
        return max_nights, violations, balance_penalty
    
    def get_neighborhood_move(self, X, move_type):
        """Generate a neighbor solution by applying a move"""
        X_new = np.copy(X)
        
        if move_type == 0:
            # Move 1: Swap two staff's shifts on same day
            day = random.randint(1, self.D)
            staff1 = random.randint(1, self.N)
            staff2 = random.randint(1, self.N)
            
            # Don't swap if either has day off
            if day in self.days_off[staff1] or day in self.days_off[staff2]:
                return X_new
                
            # Don't swap if creates night shift + work next day violation
            if (X[staff1, day] == 4 and day < self.D and staff2 in self.days_off[day+1]) or \
               (X[staff2, day] == 4 and day < self.D and staff1 in self.days_off[day+1]):
                return X_new
            
            # Perform swap
            X_new[staff1, day], X_new[staff2, day] = X_new[staff2, day], X_new[staff1, day]
            
            # Fix night shift + day off violations
            for staff in [staff1, staff2]:
                if X_new[staff, day] == 4 and day < self.D:
                    X_new[staff, day+1] = 0
            
        elif move_type == 1:
            # Move 2: Change staff's shift on specific day
            staff = random.randint(1, self.N)
            day = random.randint(1, self.D)
            
            # Skip if day off
            if day in self.days_off[staff]:
                return X_new
                
            # Skip if day after night shift
            if day > 1 and X[staff, day-1] == 4:
                return X_new
            
            # Get valid shifts
            valid_shifts = [0, 1, 2, 3]
            
            # Can work night shift only if next day is not a day off
            if day < self.D and day+1 not in self.days_off[staff]:
                valid_shifts.append(4)
            
            # Choose new shift
            new_shift = random.choice(valid_shifts)
            X_new[staff, day] = new_shift
            
            # If night shift, ensure next day is off
            if new_shift == 4 and day < self.D:
                X_new[staff, day+1] = 0
                
        elif move_type == 2:
            # Move 3: Focus on reducing max night shifts - try to redistribute
            # Find staff with most and fewest night shifts
            night_shifts = [np.sum(X[i] == 4) for i in range(1, self.N + 1)]
            max_nights = max(night_shifts) if night_shifts else 0
            min_nights = min(night_shifts) if night_shifts else 0
            
            # Only proceed if there's a difference
            if max_nights <= min_nights + 1:
                return X_new
                
            # Find staff with most night shifts
            max_staff = [i for i, n in enumerate(night_shifts, 1) if n == max_nights]
            min_staff = [i for i, n in enumerate(night_shifts, 1) if n == min_nights]
            
            if not max_staff or not min_staff:
                return X_new
                
            # Select a staff from each group
            staff_high = random.choice(max_staff)
            staff_low = random.choice(min_staff)
            
            # Find a night shift from high staff that low staff can take
            night_days = [d for d in range(1, self.D + 1) if X[staff_high, d] == 4]
            
            for day in night_days:
                # Check if low staff can take this night shift
                if (day not in self.days_off[staff_low] and 
                    (day == 1 or X[staff_low, day-1] != 4) and
                    (day == self.D or day+1 not in self.days_off[staff_low])):
                    
                    # Swap the shifts
                    X_new[staff_low, day] = 4
                    X_new[staff_high, day] = 0
                    
                    # Ensure next day is off for staff_low
                    if day < self.D:
                        X_new[staff_low, day+1] = 0
                    
                    break
        
        return X_new
    
    def simulated_annealing(self, time_limit=10):
        """Solve using simulated annealing"""
        start_time = time.time()
        
        # Initial solution
        current = self.initialize_solution()
        current_max_nights, current_violations, current_balance = self.evaluate_solution(current)
        
        # Track best solution
        best = np.copy(current)
        best_max_nights = current_max_nights
        best_violations = current_violations
        
        # SA parameters
        temp = 10.0
        cooling_rate = 0.99
        min_temp = 0.01
        iterations_per_temp = max(50, min(100, self.N))
        
        # Main SA loop
        iteration = 0
        while time.time() - start_time < time_limit and temp > min_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor
                move_type = random.choices([0, 1, 2], weights=[0.3, 0.3, 0.4])[0]
                neighbor = self.get_neighborhood_move(current, move_type)
                
                # Quick fix for violations
                if random.random() < 0.5:
                    neighbor = self.repair_solution(neighbor)
                
                # Evaluate neighbor
                neighbor_max_nights, neighbor_violations, neighbor_balance = self.evaluate_solution(neighbor)
                
                # Calculate acceptance probability
                # First prioritize feasibility
                if current_violations > 0 and neighbor_violations < current_violations:
                    accept = True
                elif current_violations == 0 and neighbor_violations == 0:
                    # If both feasible, compare by objectives
                    if neighbor_max_nights < current_max_nights:
                        accept = True
                    elif neighbor_max_nights == current_max_nights:
                        # If max nights equal, consider balance
                        if neighbor_balance < current_balance:
                            accept = True
                        else:
                            # SA acceptance criterion
                            delta = (neighbor_max_nights - current_max_nights) * 100 + (neighbor_balance - current_balance)
                            accept_prob = math.exp(-delta / temp)
                            accept = random.random() < accept_prob
                    else:
                        # SA acceptance criterion for worse max nights
                        delta = (neighbor_max_nights - current_max_nights) * 100
                        accept_prob = math.exp(-delta / temp)
                        accept = random.random() < accept_prob
                else:
                    # Compare violations
                    if neighbor_violations < current_violations:
                        accept = True
                    else:
                        delta = neighbor_violations - current_violations
                        accept_prob = math.exp(-delta / temp)
                        accept = random.random() < accept_prob
                
                # Accept or reject
                if accept:
                    current = neighbor
                    current_max_nights = neighbor_max_nights
                    current_violations = neighbor_violations
                    current_balance = neighbor_balance
                
                    # Update best solution
                    if (current_violations == 0 and 
                        (best_violations > 0 or current_max_nights < best_max_nights)):
                        best = np.copy(current)
                        best_max_nights = current_max_nights
                        best_violations = current_violations
                        
                        # Log progress
                        print(f"Iteration {iteration}: Found solution with max nights = {best_max_nights}", file=sys.stderr)
                        
                        # If we reached theoretical minimum, we can stop
                        if best_max_nights <= self.theoretical_min:
                            print(f"Reached theoretical minimum of {self.theoretical_min} night shifts", file=sys.stderr)
                            return best
                
                iteration += 1
            
            # Cool down
            temp *= cooling_rate
            
            # Check time
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.9:  # 90% of time limit
                print(f"Time limit approaching: {elapsed:.2f}s / {time_limit}s", file=sys.stderr)
                break
        
        # Final repair to ensure feasibility
        if best_violations > 0:
            best = self.repair_solution(best)
            best_max_nights, best_violations, _ = self.evaluate_solution(best)
        
        print(f"Final solution: max nights = {best_max_nights}, violations = {best_violations}", file=sys.stderr)
        return best
    
    def solve(self):
        """Main solving method"""
        # Set time limit based on problem size
        time_limit = min(15, max(5, self.N * self.D / 500))
        print(f"Using time limit: {time_limit:.2f}s", file=sys.stderr)
        
        # Run simulated annealing
        solution = self.simulated_annealing(time_limit)
        
        # Check if solution is feasible
        max_nights, violations, _ = self.evaluate_solution(solution)
        
        if violations > 0:
            print(f"Warning: Solution has {violations} violations", file=sys.stderr)
            # Try to repair one more time
            solution = self.repair_solution(solution)
            max_nights, violations, _ = self.evaluate_solution(solution)
            
            if violations > 0:
                print(f"Warning: Could not find feasible solution", file=sys.stderr)
                return None
        
        print(f"Solution quality: max night shifts = {max_nights}", file=sys.stderr)
        return solution

def main():
    start_time = time.time()
    
    # Parse input
    line = input().strip().split()
    N, D, A, B = map(int, line)
    
    days_off = {i: [] for i in range(1, N+1)}
    for i in range(1, N+1):
        days = list(map(int, input().split()))
        days_off[i] = [d for d in days if d != -1]
    
    # Solve the problem
    solver = StaffRosteringSolver(N, D, A, B, days_off)
    solution = solver.solve()
    
    if solution is None:
        print("No solution found.")
    else:
        # Print the solution
        result_matrix = np.zeros((N, D), dtype=int)
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                result_matrix[i-1][d-1] = solution[i][d]
        
        for i in range(N):
            print(" ".join(map(str, result_matrix[i])))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
