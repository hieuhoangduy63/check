from ortools.sat.python import cp_model
import numpy as np
import time
import sys

def solve_staff_rostering_binary_search(N, D, A, B, days_off):
    """Solve the staff rostering problem using binary search on the max night shifts"""
    
    # Calculate theoretical bounds
    min_max_night_shifts = max(1, (D * A + N - 1) // (N * 2))  # Ceiling division for minimum
    max_max_night_shifts = D // 2  # Maximum possible (since night + next day off)
    
    print(f"Binary search bounds: [{min_max_night_shifts}, {max_max_night_shifts}]", file=sys.stderr)
    
    # Binary search
    left = min_max_night_shifts
    right = max_max_night_shifts
    best_solution = None
    best_max_nights = float('inf')
    
    while left <= right:
        mid = (left + right) // 2
        print(f"Trying max night shifts = {mid}", file=sys.stderr)
        
        solution = solve_with_cp_sat(N, D, A, B, days_off, mid)
        
        if solution is not None:
            # Found a solution, try to improve (reduce max night shifts)
            best_solution = solution
            best_max_nights = mid
            right = mid - 1
            print(f"Found solution with max night shifts = {mid}", file=sys.stderr)
        else:
            # No solution with this limit, try higher limit
            left = mid + 1
            print(f"No solution with max night shifts = {mid}", file=sys.stderr)
    
    return best_solution

def solve_with_cp_sat(N, D, A, B, days_off, max_night_shifts):
    """Solve the staff rostering problem with CP-SAT with fixed max night shifts"""
    model = cp_model.CpModel()
    
    # Create variables
    # x[i,d,s] = 1 if staff i works shift s on day d
    x = {}
    for i in range(1, N+1):
        for d in range(1, D+1):
            for s in range(0, 5):  # 0: off, 1-4: shifts
                x[i, d, s] = model.NewBoolVar(f'x_{i}_{d}_{s}')
    
    # Constraint: each staff works exactly one shift (including off) per day
    for i in range(1, N+1):
        for d in range(1, D+1):
            model.Add(sum(x[i, d, s] for s in range(0, 5)) == 1)
    
    # Constraint: staff cannot work on their days off
    for i in range(1, N+1):
        for d in days_off[i]:
            for s in range(1, 5):
                model.Add(x[i, d, s] == 0)
    
    # Constraint: staff who work night shift (s=4) on day d must have day off on d+1
    for i in range(1, N+1):
        for d in range(1, D):
            model.Add(x[i, d+1, 0] >= x[i, d, 4])
    
    # Constraint: Each shift has between A and B staff
    for d in range(1, D+1):
        for s in range(1, 5):
            model.Add(sum(x[i, d, s] for i in range(1, N+1)) >= A)
            model.Add(sum(x[i, d, s] for i in range(1, N+1)) <= B)
    
    # Constraint: Limit night shifts to max_night_shifts for each staff
    for i in range(1, N+1):
        model.Add(sum(x[i, d, 4] for d in range(1, D+1)) <= max_night_shifts)
    
    # Secondary objective: balance the distribution of night shifts
    # Instead of minimizing total, minimize variance indirectly
    night_shift_diff = []
    for i in range(1, N+1):
        night_shifts_i = sum(x[i, d, 4] for d in range(1, D+1))
        
        # Add soft constraints to make night shifts more balanced
        # by minimizing the difference to the target value
        target = max_night_shifts - 1 if max_night_shifts > 1 else max_night_shifts
        
        # Add a bonus for night shifts less than target
        model.Maximize(50 * sum(x[i, d, s] for d in range(1, D+1) for s in range(1, 4)) - 
                      100 * sum(x[i, d, 4] for d in range(1, D+1)))
    
    # Solver settings for fast feasibility
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60  # 1-minute time limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 8
    
    # Add solution hints - distribute night shifts evenly
    avg_nights = max_night_shifts - 1 if max_night_shifts > 1 else max_night_shifts
    for i in range(1, N+1):
        night_count = 0
        for d in range(1, D+1):
            if d % N == i % N and night_count < avg_nights and d not in days_off[i]:
                if d < D and d+1 not in days_off[i]:
                    model.AddHint(x[i, d, 4], 1)  # Night shift hint
                    model.AddHint(x[i, d+1, 0], 1)  # Next day off hint
                    night_count += 1
                    d += 1  # Skip next day as it must be off
    
    # Solve
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Convert solution to matrix format
        solution = np.zeros((N, D), dtype=int)
        for i in range(1, N+1):
            for d in range(1, D+1):
                for s in range(0, 5):
                    if solver.Value(x[i, d, s]) == 1:
                        solution[i-1][d-1] = s
                        break
        return solution
    else:
        return None

def main():
    start_time = time.time()
    
    # Parse input
    line = input().strip().split()
    N, D, A, B = map(int, line)
    
    days_off = {i: [] for i in range(1, N+1)}
    for i in range(1, N+1):
        days = list(map(int, input().strip().split()))
        days_off[i] = [d for d in days if d != -1]
    
    # Try hybrid binary search approach which is much faster
    solution = solve_staff_rostering_binary_search(N, D, A, B, days_off)
    
    if solution is None:
        print("No solution found with binary search. Attempting CP-SAT with time limit...", file=sys.stderr)
        # Fallback to standard CP-SAT with objective
        from test import solve_staff_rostering
        solution = solve_staff_rostering(N, D, A, B, days_off)
    
    if solution is None:
        print("No solution found.")
    else:
        # Print the solution
        for i in range(N):
            print(" ".join(map(str, solution[i])))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds", file=sys.stderr)

if __name__ == "__main__":
    main()
