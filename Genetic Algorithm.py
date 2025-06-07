import numpy as np
import time

class GA_C:
    def __init__(self):
        pass

    def read_input(self):
        """Đọc input theo format yêu cầu"""
        # Đọc dòng đầu tiên: N, D, A, B
        line = input().strip().split()
        N, D, A, B = map(int, line)
        
        # Khởi tạo ma trận dayoff
        dayoff = np.zeros((N+1, D+1), dtype=int)
        
        # Đọc các ngày nghỉ phép của từng nhân viên
        for i in range(1, N+1):
            line = input().strip().split()
            # Chuyển đổi thành số nguyên và loại bỏ -1
            days_off = [int(x) for x in line if int(x) != -1]
            for day in days_off:
                if 1 <= day <= D:
                    dayoff[i][day] = 1
        
        return N, D, A, B, dayoff

    def print_solution(self, x, N, D):
        """In kết quả theo format yêu cầu"""
        for i in range(1, N+1):
            row = []
            for d in range(1, D+1):
                row.append(str(int(x[i][d])))
            print(' '.join(row))

    def initialize_random_solution(self, N, D, dayoff, A, B):
        X = np.zeros((N+1, D+1))
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j] == 1:
                    X[i][j] = 0
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j]:
                    continue
                if j> 1 and X[i][j - 1] == 4:
                    X[i][j] = 0
                else: 
                    X[i][j] = np.random.randint(0, 5)
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                while count < A: 
                    available = [i for i in range(1, N+1) if X[i][d] == 0 and dayoff[i][d] == 0 and (d == 1 or X[i][d-1] != 4)]
                    if not available:
                        break
                    i = np.random.choice(available)
                    X[i][d] = shift
                    count += 1
                while count > B:
                    assigned = np.where(X[:, d] == shift)[0]
                    if not assigned.size:
                        break
                    i = np.random.choice(assigned)
                    X[i][d] = 0
                    count -= 1
        return X
    
    def evaluate_fitness(self, X, N, D, A, B, dayoff):
        penalty = 0
        # check vi pham ngay nghi
        for i in range(1, N + 1):
            for d in range(1, D +1):
                if dayoff[i][d] and X[i][d] != 0:
                    penalty -= 1000

        # check vi pham so luong nguoi lam
        for d in range(1, D +1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < A:
                    penalty -= 100 * (A - count)
                elif count > B:
                    penalty -= 100 * (count - B)

        # check vi pham ca dem 
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    penalty -= 1000

        max_night_shift = 0
        for i in range(1, N + 1):
            count = np.sum(X[i] == 4)
            if count > max_night_shift:
                max_night_shift = count
        return -max_night_shift + penalty
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_index]
    
    def crossover(self, parent1, parent2, crossover_rate=0.6):
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        for i in range(1, len(parent1)):
            if np.random.rand() > crossover_rate:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        return child1, child2
    
    def mutate(self, X, N, D, dayoff, A, B, mutation_rate=0.1):
        X = np.copy(X)  # Tạo bản sao
        # Bước đột biến
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if np.random.rand() < mutation_rate:
                    if dayoff[i][d] == 1:
                        X[i][d] = 0
                        continue
                    if d > 1 and X[i][d - 1] == 4:
                        X[i][d] = 0
                    else:
                        X[i][d] = np.random.randint(0, 5)  # Bao gồm ca 4
                        if X[i][d] == 4 and d < D:
                            X[i][d + 1] = 0
        # Bước sửa lỗi ca đêm
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    X[i][d + 1] = 0
        # Bước điều chỉnh A <= số nhân viên mỗi ca <= B
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                while count < A:
                    available = [i for i in range(1, N + 1) if X[i][d] == 0 and dayoff[i][d] == 0 and (d == 1 or X[i][d - 1] != 4)]
                    if not available:
                        break
                    i = np.random.choice(available)
                    X[i][d] = shift
                    if shift == 4 and d < D:  # Nếu gán ca đêm, ngày tiếp theo phải nghỉ
                        X[i][d + 1] = 0
                    count += 1
                while count > B:
                    assigned = np.where(X[:, d] == shift)[0]
                    if not assigned.size:
                        break
                    i = np.random.choice(assigned)
                    X[i][d] = 0
                    count -= 1
        # Bước sửa lỗi ca đêm lần cuối
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    X[i][d + 1] = 0
        return X

    def check_solution(self, X, N, D, A, B, dayoff):
        """Kiểm tra tính hợp lệ của giải pháp"""
        if X is None:
            return False
            
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] == 1 and X[i][d] != 0:
                    return False
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < A or count > B:
                    return False
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    return False
        return True

    def solve(self, N, D, A, B, dayoff):
        starttime = time.time()
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        population = [self.initialize_random_solution(N, D, dayoff, A, B) for _ in range(population_size)]

        best_solution = None
        best_fitness = float('-inf')

        for generation in range(generations):
            fitness_scores = np.array([self.evaluate_fitness(individual, N, D, A, B, dayoff) for individual in population])
            current_best_index = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_index]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = np.copy(population[current_best_index])
            
            new_population = [population[current_best_index]]
            
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1, N, D, dayoff, A, B, mutation_rate))
                if len(new_population) < population_size:
                    new_population.append(self.mutate(child2, N, D, dayoff, A, B, mutation_rate))

            population = new_population[:population_size]

        return best_solution

def main():
    solver = GA_C()
    
    # Đọc input
    N, D, A, B, dayoff = solver.read_input()
    
    # Giải bài toán
    solution = solver.solve(N, D, A, B, dayoff)
    
    # Kiểm tra và in kết quả
    if solver.check_solution(solution, N, D, A, B, dayoff):
        solver.print_solution(solution, N, D)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()