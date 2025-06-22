#PYTHON 
import numpy as np
import time
import math
import random
import sys

class GeneticAlgorithmSolver:
    def __init__(self):
        pass
    
    def read_input(self):
        line = input().strip().split()
        N, D, A, B = map(int, line)
        
        dayoff = np.zeros((N+1, D+1), dtype=int)
        for i in range(1, N+1):
            line = input().strip().split()
            days_off = [int(x) for x in line if int(x) != -1]
            for day in days_off:
                if 1 <= day <= D:
                    dayoff[i][day] = 1
        
        return N, D, A, B, dayoff
    
    def print_solution(self, x, N, D):
        for i in range(1, N+1):
            print(' '.join(str(int(x[i][d])) for d in range(1, D+1)))
    
    def initialize_solution(self, N, D, A, B, dayoff):
        """Khởi tạo nghiệm ban đầu"""
        x = np.zeros((N+1, D+1), dtype=int)
        
        for d in range(1, D + 1):
            # Tìm nhân viên khả dụng
            available = []
            for i in range(1, N + 1):
                if dayoff[i][d] == 0 and (d == 1 or x[i, d-1] != 4):
                    available.append(i)
            
            if len(available) == 0:
                continue
            
            # Phân ca đơn giản
            shifts_needed = min(4 * A, len(available))
            random.shuffle(available)
            
            shift_idx = 0
            for i in range(shifts_needed):
                shift = (shift_idx % 4) + 1
                x[available[i], d] = shift
                if (i + 1) % A == 0:
                    shift_idx += 1
                    
                # Nếu ca đêm (ca 4), đảm bảo ngày hôm sau không làm việc
                if shift == 4 and d < D:
                    x[available[i], d+1] = 0
        
        return x
    
    def repair_solution(self, x, N, D, A, B, dayoff):
        """Sửa chữa các vi phạm ràng buộc"""
        x = np.copy(x)
        
        # Sửa vi phạm ngày nghỉ và ca đêm
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] == 1:
                    x[i, d] = 0
                if d < D and x[i, d] == 4:
                    x[i, d+1] = 0
        
        # Sửa số lượng nhân viên mỗi ca
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(x[:, d] == shift)
                
                # Thiếu nhân viên
                while count < A:
                    available = [i for i in range(1, N + 1) 
                               if x[i, d] == 0 and dayoff[i][d] == 0 and 
                               (d == 1 or x[i, d-1] != 4)]
                    if not available:
                        break
                    
                    i = min(available, key=lambda i: np.sum(x[i] > 0))  # tìm nhân viên có số ngày làm việc ít nhất
                    x[i, d] = shift
                    if shift == 4 and d < D:
                        x[i, d+1] = 0
                    count += 1
                
                # Thừa nhân viên -> tìm nhân viên có số ngày làm việc nhiều nhất và xoá ca
                while count > B:
                    assigned = [i for i in range(1, N + 1) if x[i, d] == shift]
                    if not assigned:
                        break
                    i = max(assigned, key=lambda i: np.sum(x[i] > 0))
                    x[i, d] = 0
                    count -= 1
        
        return x
    
    def evaluate_solution(self, x, N, D, A, B, dayoff):
        """Tính số ca đêm tối đa và kiểm tra ràng buộc"""
        night_shifts = [np.sum(x[i] == 4) for i in range(1, N + 1)]  # Số ca đêm của mỗi nhân viên
        max_nights = max(night_shifts) if night_shifts else 0  # số ca đêm nhiều nhất
        
        violations = 0
        # Vi phạm ngày nghỉ và ca đêm
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] == 1 and x[i][d] != 0:
                    violations += 100
                if d < D and x[i][d] == 4 and x[i][d+1] != 0:
                    violations += 100
        
        # Vi phạm số lượng nhân viên
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(x[:, d] == shift)
                if count < A:
                    violations += 10 * (A - count)
                elif count > B:
                    violations += 10 * (count - B)
        
        return max_nights, violations
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Chọn lọc giải đấu"""
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_index]
    
    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """Lai ghép hai nghiệm cha mẹ"""
        N, D = parent1.shape
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Lai ghép theo hàng 
        for i in range(1, N):
            if np.random.rand() < crossover_rate:
                child1[i], child2[i] = parent2[i], parent1[i]
                
        return child1, child2
    
    def mutate(self, x, N, D, dayoff, A, B, mutation_rate=0.05):
        """Đột biến nghiệm"""
        x_new = x.copy()
        
        # Giai đoạn đột biến - sử dụng phép toán vector hóa khi có thể
        mutation_mask = np.random.rand(N+1, D+1) < mutation_rate
        mutation_mask[0, :] = False  # Không đột biến hàng 0
        mutation_mask[:, 0] = False  # Không đột biến cột 0
        
        # Áp dụng đột biến có xem xét các ràng buộc
        for i in range(1, N+1):
            for d in range(1, D+1):
                if mutation_mask[i, d]:
                    if dayoff[i, d] == 1:
                        x_new[i, d] = 0    #ngày nghỉ phải = 0
                    elif d > 1 and x_new[i, d-1] == 4:
                        x_new[i, d] = 0    #sau ca đêm là ngày nghỉ
                    else:
                        x_new[i, d] = np.random.randint(0, 5) # Ngẫu nhiên 0-4
        
        # Sửa chữa vi phạm ca đêm
        for i in range(1, N+1):
            for d in range(1, D):
                if x_new[i, d] == 4:
                    x_new[i, d+1] = 0
        
        return x_new
    
    def solve(self, N, D, A, B, dayoff, time_limit=10):
        """Giải bằng thuật toán Genetic Algorithm"""
        start_time = time.time()
        theoretical_min = math.ceil(D * A / N)
        
        # Tham số GA
        population_size = 30  #kích thước quần thể là 30 nghiệm
        generations = 15   #số lần lặp của quá trình tiến hoá
        mutation_rate = 0.05
        crossover_rate = 0.8
        
        # Khởi tạo quần thể
        population = []
        for _ in range(population_size):
            individual = self.initialize_solution(N, D, A, B, dayoff)
            if random.random() < 0.3:  # 30% nghiệm được sửa chữa ngay từ đầu
                individual = self.repair_solution(individual, N, D, A, B, dayoff)
            population.append(individual)
        
        best = None
        best_max_nights = float('inf')
        best_violations = float('inf')
        
        for generation in range(generations):
            if time.time() - start_time > time_limit:
                break
                
            # Đánh giá quần thể
            fitness_scores = []
            for individual in population:
                max_nights, violations = self.evaluate_solution(individual, N, D, A, B, dayoff)
                
                # Tính fitness: ưu tiên giảm vi phạm trước, sau đó giảm ca đêm
                if violations == 0:
                    fitness = 1000 - max_nights  # Nghiệm hợp lệ
                else:
                    fitness = -violations  # Nghiệm không hợp lệ
                
                fitness_scores.append(fitness)
                
                # Cập nhật nghiệm tốt nhất
                if (violations == 0 and 
                    (best_violations > 0 or max_nights < best_max_nights)):
                    best = np.copy(individual)
                    best_max_nights, best_violations = max_nights, violations
                    
                    if best_max_nights <= theoretical_min:  # dừng sớm vì đạt tối ưu
                        return best
            
            fitness_scores = np.array(fitness_scores)
            
            # Ưu tú - giữ lại nghiệm tốt nhất
            best_idx = np.argmax(fitness_scores)
            new_population = [population[best_idx]]
            
            # Tạo quần thể mới
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2, crossover_rate)
                child1 = self.mutate(child1, N, D, dayoff, A, B, mutation_rate)
                
                # Sửa chữa ngẫu nhiên
                if random.random() < 0.4:
                    child1 = self.repair_solution(child1, N, D, A, B, dayoff)
                
                new_population.append(child1)
                
                if len(new_population) < population_size:
                    child2 = self.mutate(child2, N, D, dayoff, A, B, mutation_rate)
                    
                    if random.random() < 0.4:
                        child2 = self.repair_solution(child2, N, D, A, B, dayoff)
                    
                    new_population.append(child2)
            
            population = new_population[:population_size]
        
        if best is not None and best_violations > 0:
            best = self.repair_solution(best, N, D, A, B, dayoff)
        
        return best
    
    def check_solution(self, x, N, D, A, B, dayoff):
        """Kiểm tra tính hợp lệ của lời giải"""
        if x is None:
            return False
            
        # kiểm tra vi phạm ngày nghỉ
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] == 1 and x[i][d] != 0:
                    return False
                if d < D and x[i][d] == 4 and x[i][d + 1] != 0:
                    return False

        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(x[:, d] == shift)
                if count < A or count > B:
                    return False

        return True

def main():
    solver = GeneticAlgorithmSolver()
    N, D, A, B, dayoff = solver.read_input()
    
    time_limit = min(15, max(5, N * D / 500))
    solution = solver.solve(N, D, A, B, dayoff, time_limit)
    
    if solver.check_solution(solution, N, D, A, B, dayoff):
        solver.print_solution(solution, N, D)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()
