#PYTHON 
import numpy as np
import time
import math
import random
import sys

class SimulatedAnnealingSolver:
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
                    
                    i = min(available, key=lambda i: np.sum(x[i] > 0))  #tìm nhân viên có số ngày làm việc ít nhất
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
        night_shifts = [np.sum(x[i] == 4) for i in range(1, N + 1)] # Số ca đêm của mỗi nhân viên
        max_nights = max(night_shifts) if night_shifts else 0  #số ca đêm nhiều nhất
        
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
    
    def get_neighborhood_move(self, x, N, D, dayoff, current_max_nights=None):
        """Tạo nghiệm lân cận với chiến lược tập trung vào cân bằng ca đêm"""
        x_new = np.copy(x)
        
        # Tính toán thông tin hiện tại về ca đêm
        night_shifts = [np.sum(x[i] == 4) for i in range(1, N + 1)]
        if current_max_nights is None:
            current_max_nights = max(night_shifts) if night_shifts else 0
        
        # Xác định nhân viên có số ca đêm nhiều nhất và ít nhất
        max_staff = [i for i, n in enumerate(night_shifts, 1) if n == current_max_nights]
        min_staff = [i for i, n in enumerate(night_shifts, 1) if n < current_max_nights - 1]
        
        # Chiến lược 1: Luôn ưu tiên phân phối lại ca đêm nếu có thể
        if max_staff and min_staff:
            # Thử phân phối lại ca đêm từ nhân viên có nhiều ca đêm nhất
            staff_high = random.choice(max_staff)
            staff_low = random.choice(min_staff)
            
            # Tìm ngày mà nhân viên có nhiều ca đêm đang làm ca đêm
            night_days = [d for d in range(1, D + 1) if x[staff_high, d] == 4]
            
            if night_days:
                random.shuffle(night_days)  # Xáo trộn để không tập trung vào một khoảng thời gian
                
                for day in night_days:
                    # Kiểm tra xem nhân viên ít ca đêm có thể làm ca đêm vào ngày này không
                    if (dayoff[staff_low][day] == 0 and 
                        (day == 1 or x[staff_low, day-1] != 4) and
                        (day == D or dayoff[staff_low][day+1] == 0)):
                        
                        # Lưu ca hiện tại của nhân viên thấp
                        current_shift = x_new[staff_low, day]
                        
                        # Hoán đổi ca
                        x_new[staff_low, day] = 4
                        x_new[staff_high, day] = current_shift
                        
                        # Cập nhật ngày nghỉ sau ca đêm
                        if day < D:
                            x_new[staff_low, day+1] = 0
                            # Nếu nhân viên cao không còn làm ca đêm, bỏ ràng buộc nghỉ ngày hôm sau
                            if x_new[staff_high, day] != 4:
                                # Chỉ bỏ ràng buộc nếu không phải ngày nghỉ bắt buộc
                                if dayoff[staff_high][day+1] == 0:
                                    # Để trạng thái là 0, sau này repair_solution có thể gán ca khác nếu cần
                                    pass
                        
                        return x_new  # Trả về ngay khi tìm được cải thiện
                
        # Nếu không thể phân phối lại ca đêm, áp dụng các chiến lược khác
        if random.random() < 0.5:
            # Chiến lược 2: Hoán đổi ca của hai nhân viên
            day = random.randint(1, D)
            attempts = 0
            while attempts < 10:  # Giới hạn số lần thử
                staff1, staff2 = random.randint(1, N), random.randint(1, N)
                if staff1 != staff2 and dayoff[staff1][day] == 0 and dayoff[staff2][day] == 0:
                    # Kiểm tra ràng buộc ca đêm của ngày trước
                    valid_swap = True
                    for staff in [staff1, staff2]:
                        if day > 1 and x[staff, day-1] == 4:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Thực hiện hoán đổi
                        x_new[staff1, day], x_new[staff2, day] = x_new[staff2, day], x_new[staff1, day]
                        
                        # Xử lý ràng buộc ca đêm
                        for staff in [staff1, staff2]:
                            if x_new[staff, day] == 4 and day < D:
                                x_new[staff, day+1] = 0
                            elif x[staff, day] == 4 and day < D and x_new[staff, day] != 4:
                                # Nếu nhân viên không còn làm ca đêm
                                if dayoff[staff][day+1] == 0:
                                    # Có thể để ngày hôm sau là 0
                                    pass
                        
                        break
                attempts += 1
        else:
            # Chiến lược 3: Thay đổi ca của nhân viên
            attempts = 0
            while attempts < 10:
                staff = random.randint(1, N)
                day = random.randint(1, D)
                
                if dayoff[staff][day] == 0 and (day == 1 or x[staff, day-1] != 4):
                    # Xác định các ca có thể gán
                    valid_shifts = [0, 1, 2, 3]
                    
                    # Thêm ca đêm nếu phù hợp
                    if day < D and dayoff[staff][day+1] == 0:
                        valid_shifts.append(4)
                    
                    # Loại bỏ ca hiện tại để đảm bảo có sự thay đổi
                    current_shift = x[staff, day]
                    if current_shift in valid_shifts:
                        valid_shifts.remove(current_shift)
                    
                    if valid_shifts:
                        new_shift = random.choice(valid_shifts)
                        x_new[staff, day] = new_shift
                        
                        # Xử lý ràng buộc ca đêm
                        if new_shift == 4 and day < D:
                            x_new[staff, day+1] = 0
                        elif current_shift == 4 and day < D:
                            # Nếu trước đó là ca đêm nhưng giờ không phải
                            if dayoff[staff][day+1] == 0:
                                # Có thể mở ràng buộc ngày nghỉ hôm sau
                                pass
                        
                        break
                attempts += 1
        
        return x_new
    
    def solve(self, N, D, A, B, dayoff, time_limit=10):
        """Giải bằng thuật toán Simulated Annealing"""
        start_time = time.time()
        theoretical_min = math.ceil(D * A / N)
        
        # Nghiệm ban đầu
        current = self.initialize_solution(N, D, A, B, dayoff)
        current_max_nights, current_violations = self.evaluate_solution(current, N, D, A, B, dayoff)
        
        best = np.copy(current)
        best_max_nights, best_violations = current_max_nights, current_violations
        
        # Tham số SA
        temp = 10.0                                     #nhiệt độ ban đầu
        cooling_rate = 0.99                        #tốc độ làm mát (giảm nhiệt độ 1% mỗi chu kỳ)
        iterations_per_temp = max(50, min(100, N))        #số lần lặp mỗi nhiệt độ
        
        while time.time() - start_time < time_limit and temp > 0.01:
            for _ in range(iterations_per_temp):
                # Tạo và đánh giá nghiệm lân cận với hàm mới
                neighbor = self.get_neighborhood_move(current, N, D, dayoff, current_max_nights)
                
                #sửa chữa ngẫu nhiên rồi đánh giá nghiệm mới
                if random.random() < 0.5:
                    neighbor = self.repair_solution(neighbor, N, D, A, B, dayoff)
                
                neighbor_max_nights, neighbor_violations = self.evaluate_solution(neighbor, N, D, A, B, dayoff)
                
                # Tính xác suất chấp nhận: so sánh nghiệm hiện tại và nghiệm lân cận
                accept = False
                if current_violations > 0 and neighbor_violations < current_violations:
                    accept = True
                elif current_violations == 0 and neighbor_violations == 0:
                    if neighbor_max_nights <= current_max_nights:
                        accept = True
                    else:
                        delta = (neighbor_max_nights - current_max_nights) * 100   #*100 để tăng penalty, khó chấp nhận nghiệm xấu hơn
                        accept = random.random() < math.exp(-delta / temp)         # SA probability
                else: # neighbor_violations >= current_violations
                    delta = neighbor_violations - current_violations       #delta cho vi phạm ràng buộc nhỏ hơn để ưu tiên giải quyết
                    accept = random.random() < math.exp(-delta / temp)
                
                if accept:
                    current = neighbor
                    current_max_nights, current_violations = neighbor_max_nights, neighbor_violations
                    
                    if (current_violations == 0 and 
                        (best_violations > 0 or current_max_nights < best_max_nights)):
                        best = np.copy(current)
                        best_max_nights, best_violations = current_max_nights, current_violations
                        
                        if best_max_nights <= theoretical_min:  #dừng sớm vì đạt tối ưu
                            return best
            
            temp *= cooling_rate
        
        if best_violations > 0:
            best = self.repair_solution(best, N, D, A, B, dayoff)
        
        return best
    
    def check_solution(self, x, N, D, A, B, dayoff):
        """Kiểm tra tính hợp lệ của lời giải"""
        if x is None:
            return False
            
        #kiểm tra vi phạm ngày nghỉ
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
    solver = SimulatedAnnealingSolver()
    N, D, A, B, dayoff = solver.read_input()
    
    time_limit = min(15, max(5, N * D / 500))
    solution = solver.solve(N, D, A, B, dayoff, time_limit)
    
    if solver.check_solution(solution, N, D, A, B, dayoff):
        solver.print_solution(solution, N, D)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()