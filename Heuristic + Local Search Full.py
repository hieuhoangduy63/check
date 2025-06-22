#PYTHON 
import numpy as np
import time
import random

class Heuristic_LocalSearch:
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
    
    def solve(self, N, D, A, B, dayoff):
        """Giải bài toán với heuristic + local search"""
        start = time.time()
        
        # Tính số ca đêm tối thiểu cần thiết
        total_night_shifts = D * A
        min_max_nights = (total_night_shifts + N - 1) // N   #(a + b - 1) // b = ceil(a / b) (làm tròn lên)
        
        # Thử từ mức tối thiểu
        for max_nights in range(min_max_nights, D + 1):
            for attempt in range(10):  # Thử nhiều lần với randomization
                solution = self.solve_with_max_nights(N, D, A, B, dayoff, max_nights, attempt)
                if solution is not None:
                    improved_solution = self.improve_solution(solution, N, D, A, B, dayoff)
                    return improved_solution
        
        return None

    def solve_with_max_nights(self, N, D, A, B, dayoff, max_nights, seed):
        """Giải với ràng buộc max_nights"""
        random.seed(seed)     # Đặt seed để tạo tính ngẫu nhiên, nếu cùng 1 seed sẽ nhận lại đúng giải pháp cũ
        
        # Khởi tạo ma trận lịch làm việc
        x = np.zeros((N+1, D+1))
        
        # Theo dõi số ca đêm của từng nhân viên
        night_count = np.zeros(N+1)
        
        # Xử lý từng ngày
        for day in range(1, D+1):
            if not self.schedule_single_day(x, N, day, A, B, dayoff, max_nights, night_count):
                return None
        
        return x

    def schedule_single_day(self, x, N, day, A, B, dayoff, max_nights, night_count):
        """Lập lịch cho một ngày"""
        # Tìm nhân viên có thể làm việc
        available_staff = self.get_available_staff(x, N, day, dayoff)
        
        # Kiểm tra tính khả thi
        if len(available_staff) < 4 * A:
            return False
        
        # Phân chia staff vào các ca
        return self.assign_shifts_to_day(x, available_staff, day, A, B, max_nights, night_count)

    def get_available_staff(self, x, N, day, dayoff):
        """Lấy danh sách nhân viên có thể làm việc"""
        available_staff = []
        for i in range(1, N+1):
            # Kiểm tra ngày nghỉ phép
            if dayoff[i][day] == 1:
                continue
            
            # Kiểm tra ca đêm ngày trước
            if day > 1 and x[i][day-1] == 4:
                continue
            
            # Kiểm tra đã được phân công chưa
            if x[i][day] != 0:
                continue
            
            available_staff.append(i)
        
        return available_staff

    def assign_shifts_to_day(self, x, available_staff, day, A, B, max_nights, night_count):
        """Phân công ca cho một ngày"""
        # Sắp xếp staff theo priority
        staff_priority = []
        for staff in available_staff:
            priority = self.calculate_staff_priority(staff, night_count, max_nights)
            staff_priority.append((staff, priority))
        
        staff_priority.sort(key=lambda x: x[1]) # Sắp xếp theo priority tăng dần (số nhỏ = priority cao)
        
        # Phân công ca đêm trước (ca 4)
        night_candidates = [s for s, p in staff_priority               #danh sách những nhân viên chưa vượt quá max_nights
                           if night_count[s] < max_nights]
        
        if len(night_candidates) < A:
            return False
        
        # Chọn A nhân viên có priority cao nhất cho ca đêm
        assigned_staff = set()         #danh sách để lưu các nhân viên được phân ca rong ngày
        for i in range(A):
            staff = night_candidates[i]
            x[staff][day] = 4
            night_count[staff] += 1
            assigned_staff.add(staff)
        
        # Phân công các ca còn lại (ca 1, 2, 3)
        remaining_staff = [s for s, p in staff_priority if s not in assigned_staff]
        random.shuffle(remaining_staff)  # Tạo tính ngẫu nhiên
        
        shift_index = 0
        for shift in range(1, 4):
            if len(remaining_staff) < A:
                return False
            
            for i in range(A):
                if shift_index >= len(remaining_staff):
                    return False
                
                staff = remaining_staff[shift_index]
                x[staff][day] = shift
                assigned_staff.add(staff)
                shift_index += 1
        
        return True

    def calculate_staff_priority(self, staff, night_count, max_nights):
        """Tính priority của nhân viên (số nhỏ = priority cao)"""
        # Ưu tiên nhân viên có ít ca đêm hơn
        priority = night_count[staff]
        
        # Nếu đã đạt max_nights thì priority thấp
        if night_count[staff] >= max_nights:
            priority += 1000
        
        return priority

    def improve_solution(self, x, N, D, A, B, dayoff):
        """Cải thiện giải pháp bằng local search"""
        best_x = x.copy()
        best_max_nights = self.get_max_nights(x, N, D)
        
        # Thực hiện local search
        for iteration in range(200):
            # Tìm cặp (staff, day) để tối ưu
            improvements = self.find_improvement_opportunities(x, N, D, dayoff)
            
            if not improvements:
                break
            
            # Chọn ngẫu nhiên một cải tiến
            staff1, day1, staff2, day2 = random.choice(improvements)
            
            # Thực hiện swap
            temp = x[staff1][day1]
            x[staff1][day1] = x[staff2][day2]
            x[staff2][day2] = temp
            
            # Kiểm tra nếu cải thiện
            new_max_nights = self.get_max_nights(x, N, D)
            if new_max_nights < best_max_nights:
                best_max_nights = new_max_nights
                best_x = x.copy()
            else:
                # Hoàn tác nếu không cải thiện
                temp = x[staff1][day1]
                x[staff1][day1] = x[staff2][day2]
                x[staff2][day2] = temp
        
        # Áp dụng kết quả tốt nhất
        return best_x

    def find_improvement_opportunities(self, x, N, D, dayoff):
        """Tìm cơ hội cải thiện"""
        improvements = []
        night_counts = [np.sum(x[i, :] == 4) for i in range(1, N+1)]    #số ca đêm của từng nhân viên
        
        # Tìm nhân viên có nhiều ca đêm nhất
        max_nights = max(night_counts)
        max_staff_list = [i+1 for i in range(N) if night_counts[i] == max_nights]   #danh sách các nhân viên có số ca đêm max
        
        for staff1 in max_staff_list:
            for day1 in range(1, D+1):
                if x[staff1][day1] == 4:  # Ca đêm
                    # Tìm nhân viên có thể thay thế
                    for staff2 in range(1, N+1):
                        if (staff2 != staff1 and 
                            night_counts[staff2-1] < night_counts[staff1-1] and
                            self.can_swap_shifts(x, staff1, day1, staff2, day1, dayoff)):
                            improvements.append((staff1, day1, staff2, day1))
        
        return improvements

    def can_swap_shifts(self, x, staff1, day1, staff2, day2, dayoff):
        """Kiểm tra có thể swap ca không"""
        # Kiểm tra ngày nghỉ phép
        if dayoff[staff2][day1] == 1 or dayoff[staff1][day2] == 1:
            return False
        
        # Kiểm tra ràng buộc ca đêm
        if x[staff1][day1] == 4:  # staff1 làm ca đêm
            # staff2 có thể làm ca đêm không?
            if day1 > 1 and x[staff2][day1-1] == 4:
                return False
            if day1 < x.shape[1]-1 and x[staff2][day1+1] != 0:  #nếu day1 không phải ngày cuối
                return False
        
        if x[staff2][day2] == 4:  # staff2 làm ca đêm
            # staff1 có thể làm ca đêm không?
            if day2 > 1 and x[staff1][day2-1] == 4:
                return False
            if day2 < x.shape[1]-1 and x[staff1][day2+1] != 0:
                return False
        
        return True

    def get_max_nights(self, x, N, D):
        """Tính số ca đêm tối đa"""
        max_nights = 0
        for i in range(1, N+1):
            nights = np.sum(x[i, :] == 4)
            max_nights = max(max_nights, int(nights))
        return max_nights

    def check_solution(self, x, N, D, A, B, dayoff):
        """Kiểm tra tính hợp lệ của lời giải"""
        if x is None:
            return False
            
        # Kiểm tra vi phạm ngày nghỉ
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] and x[i][d] != 0:
                    return False

        # Kiểm tra vi phạm số lượng nhân viên mỗi ca
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(x[:, d] == shift)
                if count < A or count > B:
                    return False

        # Kiểm tra vi phạm ca đêm
        for i in range(1, N + 1):
            for d in range(1, D):
                if x[i][d] == 4 and x[i][d + 1] != 0:
                    return False

        return True

def main():
    solver = Heuristic_LocalSearch()
    
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
