from ortools.sat.python import cp_model
import numpy as np

def read_input():
    N, D, A, B = map(int, input().split())
    dayoff = np.zeros((N, D), dtype=int)
    for i in range(N):
        days = list(map(int, input().split()))
        for d in days:
            if d == -1:
                break
            if 1 <= d <= D:
                dayoff[i][d - 1] = 1
    return N, D, A, B, dayoff

def print_solution(schedule, N, D):
    for i in range(N):
        print(" ".join(str(schedule[i][d]) for d in range(D)))

def solve_schedule_cp(N, D, A, B, dayoff):
    model = cp_model.CpModel()

    # x[i][d][s] = 1 if staff i works shift s on day d (s = 0 to 3 for shifts 1 to 4)
    x = {}
    for i in range(N):
        for d in range(D):
            for s in range(4):
                x[i, d, s] = model.NewBoolVar(f'x_{i}_{d}_{s}')

    # Each staff works at most 1 shift per day
    for i in range(N):
        for d in range(D):
            model.AddAtMostOne(x[i, d, s] for s in range(4))

    # Respect day-off constraints
    for i in range(N):
        for d in range(D):
            if dayoff[i][d] == 1:
                for s in range(4):
                    model.Add(x[i, d, s] == 0)

    # No shift the day after a night shift (shift 4)
    for i in range(N):
        for d in range(D - 1):
            model.Add(x[i, d, 3] + sum(x[i, d + 1, s] for s in range(4)) <= 1)

    # For each shift each day, enforce staff count in [A, B]
    for d in range(D):
        for s in range(4):
            shift_count = sum(x[i, d, s] for i in range(N))
            model.Add(shift_count >= A)
            model.Add(shift_count <= B)

    # Optional: balance night shifts (shift 4)
    max_night = model.NewIntVar(0, D, 'max_night')
    for i in range(N):
        night_shifts = sum(x[i, d, 3] for d in range(D))
        model.Add(night_shifts <= max_night)
    model.Minimize(max_night)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = np.zeros((N, D), dtype=int)
        for i in range(N):
            for d in range(D):
                for s in range(4):
                    if solver.Value(x[i, d, s]):
                        schedule[i][d] = s + 1  # Shifts 1 to 4
        return schedule
    else:
        return None

def main():
    N, D, A, B, dayoff = read_input()
    solution = solve_schedule_cp(N, D, A, B, dayoff)
    if solution is not None:
        print_solution(solution, N, D)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()
