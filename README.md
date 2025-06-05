# Staff Rostering Problem - Minimize Maximum Night Shifts

*[Tiếng Việt](#tiếng-việt) | [English](#english)*

---

## English

### Problem Description

This project solves a staff rostering optimization problem where N employees need to be scheduled for work shifts across D days. The goal is to minimize the maximum number of night shifts assigned to any single employee.

### Problem Statement

Given:
- **N employees** (indexed 1, 2, ..., N)
- **D days** (indexed 1, 2, ..., D) 
- Each day has **4 shifts**: Morning (1), Afternoon (2), Evening (3), Night (4)
- **Day-off** is represented by (0)

### Constraints

1. **Single shift per day**: Each employee works at most one shift per day
2. **Night shift rest rule**: If an employee works the night shift, they must have the next day off
3. **Shift capacity**: Each shift on each day must have at least **A** employees and at most **B** employees
4. **Scheduled leave**: Each employee has predefined days off F(i)

### Objective

Minimize the maximum number of night shifts assigned to any single employee across all D days.

### Input Format

```
Line 1: N D A B
Lines 2 to N+1: For each employee i, list of their scheduled days off, terminated by -1
```

Where:
- `N`: Number of employees (1 ≤ N ≤ 500)
- `D`: Number of days (1 ≤ D ≤ 200)  
- `A`: Minimum employees per shift (1 ≤ A ≤ B ≤ 500)
- `B`: Maximum employees per shift
- Days are indexed from 1 to D

### Output Format

```
N lines: Each line i contains the shift schedule for employee i across D days
```

Where each value represents:
- `0`: Day off
- `1`: Morning shift
- `2`: Afternoon shift  
- `3`: Evening shift
- `4`: Night shift

### Example

#### Input
```
8 6 1 3
1 -1
3 -1
4 -1
5 -1
2 4 -1
-1
-1
3 -1
```

#### Output
```
0 1 3 1 4 0
4 0 0 1 2 2
2 4 0 0 2 2
3 1 4 0 0 4
1 0 2 0 1 1
3 2 1 2 3 3
2 3 2 4 0 3
1 3 0 3 1 1
```

---

## Tiếng Việt

### Staff Rostering Problem Minimize Max Night Shift Ext 1

### Description

Có N nhân viên 1,2,…, N cần được xếp ca trực làm việc cho các ngày 1, 2, …, D. Mỗi ngày được chia thành 4 kíp: sáng, trưa, chiều, đêm. Biết rằng:

* Mỗi ngày, một nhân viên chỉ làm nhiều nhất 1 ca
* Ngày hôm trước làm ca đêm thì hôm sau được nghỉ
* Mỗi ca trong mỗi ngày có ít nhất A nhân viên và nhiều nhất B nhân viên
* F(i): danh sách các ngày nghỉ phép của nhân viên i

Xây dựng phương án xếp ca trực cho N nhân viên sao cho:
* Số ca đêm nhiều nhất phân cho 1 nhân viên nào đó là nhỏ nhất

A solution is represented by a matrix X[1..N][1..D] in which x[i][d] is the shift scheduled to staff i on day d (value 1 means shift morning; value 2 means shift afternoon; value 3 means shift evening; value 4 means shift night; value 0 means day-off)

### Input

* Line 1: contains 4 positive integers N, D, A, B (1 <= N <= 500, 1 <= D <= 200, 1 <= A <= B <= 500)
* Line i + 1 (i = 1, 2, . . ., N): contains a list of positive integers which are the day off of the staff i (days are indexed from 1 to D), terminated by -1

### Output

* Line i (i = 1, 2, . . ., N): write the ith row of the solution matrix X

### Example

#### Input
```
8 6 1 3
1 -1
3 -1
4 -1
5 -1
2 4 -1
-1
-1
3 -1
```

#### Output
```
0 1 3 1 4 0
4 0 0 1 2 2
2 4 0 0 2 2
3 1 4 0 0 4
1 0 2 0 1 1
3 2 1 2 3 3
2 3 2 4 0 3
1 3 0 3 1 1
```
