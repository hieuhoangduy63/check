import numpy as np
from ortools.linear_solver import pywraplp

class Linear:
    def __init__(self):
        pass

    def solve(self, N, D, A, B, dayoff):
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Tao bien rang buoc
        x = {}
        for i in range(1, N+1):
            for j in range(1, D+1):
                for k in range(0, 5):
                    x[i, j, k] = solver.IntVar(0, 1, f'x[{i},{j},{k}]')

        # Rang buoc: moi nhan vien moi ngay lam nhieu nhat 1 ca
        for i in range(1, N+1):
            for j in range(1, D+1):
                solver.Add(sum(x[i, j, k] for k in range(0, 5)) == 1)  # Phải bằng 1 (hoặc nghỉ hoặc 1 ca)

        # Rang buoc neu ngay hom truoc lam ca dem (ca 4) thi ngay hom sau phai nghi
        for i in range(1, N+1):
            for j in range(1, D):
                solver.Add(x[i, j, 4] + sum(x[i, j+1, k] for k in range(1, 5)) <= 1)

        # Moi ca trong ngay co it nhat A nhan vien va nhieu nhat B nhan vien
        for j in range(1, D+1):  # cac ngay
            for k in range(1, 5):  # cac ca
                solver.Add(sum(x[i, j, k] for i in range(1, N+1)) >= A)
                solver.Add(sum(x[i, j, k] for i in range(1, N+1)) <= B)

        # Rang buoc ngay nghi cho nhan vien
        for i in range(1, N+1):
            for j in dayoff[i]:  # dayoff[i] chứa danh sách các ngày nghỉ của nhân viên i
                solver.Add(x[i, j, 0] == 1)  # Bắt buộc nghỉ

        # Bien muc tieu
        goal = solver.IntVar(0, D, 'goal')
        for i in range(1, N+1):
            solver.Add(goal >= sum(x[i, j, 4] for j in range(1, D+1)))
        
        solver.Minimize(goal)
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            best_solution = np.zeros((N+1, D+1), dtype=int)
            for i in range(1, N+1):
                for j in range(1, D+1):
                    for k in range(0, 5):
                        if x[i, j, k].solution_value() == 1:
                            best_solution[i][j] = k
            return best_solution
        else:
            return None

def read_input():
    line = input().strip().split()
    N, D, A, B = map(int, line)
    
    dayoff = {}
    for i in range(1, N+1):
        days = list(map(int, input().strip().split()))
        dayoff[i] = set()
        for day in days:
            if day == -1:
                break
            dayoff[i].add(day)
    
    return N, D, A, B, dayoff

def main():
    N, D, A, B, dayoff = read_input()
    
    solution = linear_solver.solve(N, D, A, B, dayoff)
  
    
    if solution is not None:
        # In kết quả theo định dạng yêu cầu
        for i in range(1, N+1):
            row = []
            for j in range(1, D+1):
                row.append(str(solution[i][j]))
            print(' '.join(row))
    else:
        print("No solution found")

if __name__ == "__main__":
    main()
