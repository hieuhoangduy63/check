#PYTHON 
def solve_staff_rostering(N, D, A, B, leave_days):
    # Initialize the solution matrix with all zeros (days off)
    solution = [[0 for _ in range(D)] for _ in range(N)]
    
    # Track the number of night shifts assigned to each staff member
    night_shifts_count = [0 for _ in range(N)]
    
    # Track staff assigned to each shift on each day
    shift_assignments = [[[] for _ in range(5)] for _ in range(D)]  # 5 slots for shifts 0-4
    
    # Mark leave days as unavailable
    for staff in range(N):
        for day in leave_days[staff]:
            solution[staff][day-1] = 0  # Mark as day off
    
    # First, assign night shifts strategically to minimize maximum night shifts
    for day in range(D):
        # Sort staff by number of night shifts and availability for night shift
        eligible_staff = []
        for staff in range(N):
            # Check if staff can work on this day
            if day+1 in leave_days[staff]:
                continue
                
            # Check if staff worked night shift on previous day
            if day > 0 and solution[staff][day-1] == 4:
                solution[staff][day] = 0  # Force day off after night shift
                continue
                
            # Check if assigning night shift would force unavailability on a leave day
            if day < D-1 and day+2 in leave_days[staff]:
                # Can't assign night shift as it would force day off on a leave day
                continue
                
            eligible_staff.append((night_shifts_count[staff], staff))
        
        # Sort by night shift count (ascending)
        eligible_staff.sort()
        
        # Assign night shifts (minimum A, maximum B staff)
        night_shift_assigned = 0
        for _, staff in eligible_staff:
            if night_shift_assigned >= B:
                break
                
            if night_shift_assigned < A or (night_shift_assigned < B and night_shifts_count[staff] <= min(night_shifts_count) + 1):
                solution[staff][day] = 4  # Assign night shift
                night_shifts_count[staff] += 1
                shift_assignments[day][4].append(staff)
                night_shift_assigned += 1
            
            if night_shift_assigned >= A and all(count > 0 for count in night_shifts_count if count > 0):
                # If we've assigned minimum required and balanced the load, stop
                break
    
    # Next, assign remaining shifts (morning, afternoon, evening)
    for day in range(D):
        for shift in [1, 2, 3]:  # Morning, afternoon, evening
            # Count currently assigned staff to this shift
            assigned_count = sum(1 for staff in range(N) if solution[staff][day] == shift)
            
            # Need to assign more staff to reach minimum A
            while assigned_count < A:
                # Find available staff with least shifts of this type
                best_staff = -1
                min_shifts = float('inf')
                
                for staff in range(N):
                    # Skip if already assigned or unavailable
                    if solution[staff][day] != 0:
                        continue
                    if day+1 in leave_days[staff]:
                        continue
                    if day > 0 and solution[staff][day-1] == 4:
                        continue
                        
                    # Count how many of this shift type the staff already has
                    shift_count = sum(1 for d in range(D) if solution[staff][d] == shift)
                    
                    if shift_count < min_shifts:
                        min_shifts = shift_count
                        best_staff = staff
                
                if best_staff == -1:
                    # No available staff found for this shift
                    break
                    
                solution[best_staff][day] = shift
                assigned_count += 1
    
    # Final check and adjustments to ensure all constraints are met
    for day in range(D):
        for shift in range(1, 5):  # For each shift type
            assigned = [staff for staff in range(N) if solution[staff][day] == shift]
            
            if len(assigned) < A:
                # Need more staff for this shift
                available_staff = []
                for staff in range(N):
                    if solution[staff][day] == 0 and day+1 not in leave_days[staff]:
                        if day > 0 and solution[staff][day-1] == 4:
                            continue  # Can't work after night shift
                        available_staff.append(staff)
                
                # Assign available staff until minimum requirement met
                for staff in available_staff[:A - len(assigned)]:
                    solution[staff][day] = shift
    
    return solution

def main():
    # Parse input
    line = input().strip().split()
    N, D, A, B = map(int, line)
    
    leave_days = []
    for i in range(N):
        days = list(map(int, input().strip().split()))
        leave_days.append([d for d in days if d != -1])
    
    # Solve the problem
    solution = solve_staff_rostering(N, D, A, B, leave_days)
    
    # Output the solution
    for staff in range(N):
        print(' '.join(map(str, solution[staff])))

if __name__ == "__main__":
    main()
