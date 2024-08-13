# DAA
### 1 (A) ..  IMPLEMENT RECURSIVE ALGORITHM.


#### AIM : Present the execution time visually through a graph to help understand how the performance of the GCD calculation varies.
### algorithm 


1. **Define `gcd_recursive(a, b)`:** Compute GCD using recursion.
2. **Define `reduce_gcd(numbers)`:** Compute GCD for a list of numbers.
3. **Initialize Variables:**
   - Set `num_measurements` to 1.
   - Create an empty list `execution_times`.
4. **Collect User Input:**
   - Ask for the number of integers.
   - Collect the integers from the user.i
5. **Start Timer:** Record the start time.
6. **Calculate GCD:** Use `reduce_gcd(numbers)` with the provided numbers.
7. **Stop Timer:** Record the end time.
8. **Record Execution Time:** Calculate and store the time taken.
9. **Print Results:** Output the GCD and execution time.
10. **Plot Execution Time:**
    
### program
```
import time
import matplotlib.pyplot as plt
start_time = time.time()
def gcd_recursive(a, b):
    if b == 0:
        return a
    return gcd_recursive(b, a % b)

def reduce_gcd(numbers):
    if len(numbers) == 1:
        return numbers[0]
    return gcd_recursive(numbers[0], reduce_gcd(numbers[1:]))

# Default number of measurements set to 1
num_measurements = 1
execution_times = []

for i in range(num_measurements):
    # Get number of elements and their values from the user
    n = int(input("How many numbers do you want to calculate the GCD for? "))
    numbers = [int(input(f"Enter number {j+1}: ")) for j in range(n)]
    
    # Measure execution time
    
    result = reduce_gcd(numbers)
    execution_time = time.time() - start_time
    execution_times.append(execution_time)
    
    print(f"GCD: {result}")
    print(f"Execution time: {execution_time:.6f} seconds")

# Calculate the average execution time (for a single measurement, it's just the time itself)
average_time = execution_times[0] if execution_times else 0

# Generate x-axis values (measurement)
measurements = list(range(1, num_measurements + 1))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(measurements, execution_times, marker='o', linestyle='-', color='dodgerblue', label='Execution Time')
plt.axhline(y=average_time, color='red', linestyle='--', label='Average Time')
plt.xlabel('Measurement')
plt.ylabel('Time (seconds)')
plt.title('Execution Time of GCD Calculation')
plt.legend()
plt.grid(True)
plt.show()
```
#### RESULT:Thus, the program to implement the concept of Recursive algorithm using Python has been executed successfully.

### 1B .. IMPLEMENT NON-RECURSIVE ALGORITHM.
#### AIM : To perform the sum of user-defined integers, measure the execution time, and visualize the performance through a graph.
### Algorithm 


1. **Define `sum_of_elements(numbers)`:** Calculate the sum of the list.
2. **Initialize:** Set `num_measurements` to 1 and create an empty list for execution times.
3. **User Input:** Ask for the number of integers and collect their values.
4. **Measure Execution Time:**
   - Record start time.
   - Calculate the sum using `sum_of_elements()`.
   - Record end time and compute duration.
   - Store execution time.
   - Print the sum and execution time.
5. **Plot Execution Time:** 
   - Plot the execution time with a horizontal line for average time.
### PROGRAM 
```
import time
import matplotlib.pyplot as plt

start_time = time.time()
def sum_of_elements(numbers):
    return sum(numbers)

# Default number of measurements set to 1
num_measurements = 1
execution_times = []

for i in range(num_measurements):
    # Get number of elements and their values from the user
    n = int(input("How many numbers do you want to sum? "))
    numbers = [int(input(f"Enter number {j+1}: ")) for j in range(n)]
    
    # Measure execution time
    
    result = sum_of_elements(numbers)
    execution_time = time.time() - start_time
    execution_times.append(execution_time)
    
    print(f"Sum: {result}")
    print(f"Execution time: {execution_time:.6f} seconds")

# Calculate the average execution time (for a single measurement, it's just the time itself)
average_time = execution_times[0] if execution_times else 0

# Generate x-axis values (measurement)
measurements = list(range(1, num_measurements + 1))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(measurements, execution_times, marker='o', linestyle='-', color='dodgerblue', label='Execution Time')
plt.axhline(y=average_time, color='red', linestyle='--', label='Average Time')
plt.xlabel('Measurement')
plt.ylabel('Time (seconds)')
plt.title('Execution Time of Sum Calculation')
plt.legend()
plt.grid(True)
plt.show()
```
### RESULT: The program "Sum of Array Elements Performance Analysis" was successfully executed and OUTPUT IS  verified.
##
##
#### 2 .DIVIDE AND CONQUER STRASSEN’S MATRIX MULTIPLICATION
### Aim : To implement Strassen’s Matrix Multiplication using Divide and Conquer.
#### ALGORITHM 


1. **Define `strassen_matrix_multiplication(A, B)` Function:**
   - **`split_matrix(matrix)`**: Split matrix into four sub-matrices.
   - **`strassen(A, B)`**: 
     - Recursively compute matrix product using Strassen's method.
     - Combine results from intermediary matrices.

2. **Initialize:**
   - Set `num_measurements` to 1.
   - Create an empty list for execution times.

3. **User Input:**
   - Get matrix size `n`.
   - Collect matrix values for `A` and `B` from user.

4. **Measure Execution Time:**
   - Record start time.
   - Compute matrix multiplication using `strassen_matrix_multiplication()`.
   - Record end time and calculate duration.
   - Print matrices, result, and execution time.

5. **Plot Execution Time:**
   - Calculate average time (for one measurement, it’s the recorded time).
   - Plot execution time with a line graph and average time.
   - Label axes and title, then display the plot.
### PROGRAM 
```
import time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
def strassen_matrix_multiplication(A, B):
    def split_matrix(matrix):
        r, c = matrix.shape
        r2, c2 = r // 2, c // 2
        return (matrix[:r2, :c2], matrix[:r2, c2:], matrix[r2:, :c2], matrix[r2:, c2:])
    
    def strassen(A, B):
        if len(A) == 1:
            return A * B
        
        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)
        
        M1 = strassen(A11 + A22, B11 + B22)
        M2 = strassen(A21 + A22, B11)
        M3 = strassen(A11, B12 - B22)
        M4 = strassen(A22, B21 - B11)
        M5 = strassen(A11 + A12, B22)
        M6 = strassen(A21 - A11, B11 + B12)
        M7 = strassen(A12 - A22, B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return strassen(np.array(A), np.array(B))

num_measurements = 1
execution_times = []

for _ in range(num_measurements):
    n = int(input("Enter the size of the matrices (n for nxn matrices): "))
    
    A = np.array([[int(input(f"Enter value for A[{i+1}][{j+1}]: ")) for j in range(n)] for i in range(n)])
    B = np.array([[int(input(f"Enter value for B[{i+1}][{j+1}]: ")) for j in range(n)] for i in range(n)])
    
    
    result = strassen_matrix_multiplication(A, B)
    execution_time = time.time() - start_time
    execution_times.append(execution_time)
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Result of A x B using Strassen's Algorithm:\n{result}")
    print(f"Execution time: {execution_time:.6f} seconds")

average_time = execution_times[0] if execution_times else 0
measurements = list(range(1, num_measurements + 1))

plt.figure(figsize=(12, 6))
plt.plot(measurements, execution_times, marker='o', linestyle='-', color='dodgerblue', label='Execution Time')
plt.axhline(y=average_time, color='red', linestyle='--', label='Average Time')
plt.xlabel('Measurement')
plt.ylabel('Time (seconds)')
plt.title('Execution Time of Strassen\'s Matrix Multiplication')
plt.legend()
plt.grid(True)
plt.show()
```
#### Result:: Thus, the python program to implement Strassen’s Matrix Multiplication using Divide and Conquer has been executed successfully.

## 3... DECREASE AND CONQUER TOPOLOGICAL SORTING

### Aim : To implement Topological sorting using Decrease and Conquer.
### Algorithm

1. **Define `topological_sort(graph)` Function:**
   - **Initialize In-Degree:**
     - Create a dictionary to track the in-degree of each vertex.
     - For each vertex, count the number of incoming edges and update in-degrees.
   - **Initialize Queue:**
     - Create a queue with vertices having zero in-degrees.
   - **Process Vertices:**
     - While the queue is not empty:
       - Remove a vertex from the queue.
       - Append the vertex to the sorted list.
       - For each neighbor of the vertex, decrease its in-degree.
       - If a neighbor’s in-degree becomes zero, add it to the queue.
   - **Check for Cycles:**
     - If the sorted list contains all vertices, return it.
     - Otherwise, raise an error indicating a cycle exists.

2. **Initialize Measurement:**
   - Set `num_measurements` to 1.
   - Create an empty list `execution_times` to store the execution time for each run.

3. **Collect User Input:**
   - **Number of Vertices:**
     - Prompt the user for the number of vertices in the graph.
   - **Edges Input:**
     - Initialize an empty graph representation.
     - Prompt the user to enter the start and end vertices for each edge.
     - Update the graph structure with user-provided edges.

4. **Measure Execution Time:**
   - Record the start time.
   - Call `topological_sort(graph)` to compute the topological order.
   - Record the end time and compute the duration.
   - Append the execution time to `execution_times`.
   - Print the graph, the topological sort result, and the execution time.

5. **Plot Execution Time:**
   - **Calculate Average Time:**
     - Compute the average execution time (for a single measurement, it's the recorded time itself).
   - **Generate Plot:**
     - Plot execution times using a line graph.
     - Add a horizontal line for average time.
     - Label the x-axis, y-axis, and add a title.
     - Display the plot with gridlines.
##
##
   # Easiest algorithm
1. **Define `topological_sort(graph)` Function:**
   - Compute in-degrees for each vertex.
   - Use a queue to process vertices with zero in-degrees.
   - Update in-degrees and queue as vertices are processed.
   - Check if all vertices are sorted; raise an error if not.

2. **Initialize:**
   - Set `num_measurements` to 1.
   - Create an empty list for execution times.

3. **User Input:**
   - Get the number of vertices.
   - Collect edge information to build the graph.

4. **Measure Execution Time:**
   - Record start time.
   - Perform topological sorting.
   - Record end time, compute duration, and store it.
   - Print graph, result, and execution time.

5. **Plot Execution Time:**
   - Calculate average execution time.
   - Plot execution times and average time on a graph.
   - Label the plot and display it.

## PROGRAM
```
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
start_time = time.time()
def topological_sort(graph):
    """
    Perform topological sorting using Kahn's algorithm.
    """
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in graph if in_degree[u] == 0])
    sorted_list = []

    while queue:
        u = queue.popleft()
        sorted_list.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_list) == len(graph):
        return sorted_list
    else:
        raise ValueError("Graph has at least one cycle, topological sort not possible")

num_measurements = 1
execution_times = []

for _ in range(num_measurements):
    # Get the number of vertices from user
    n = int(input("Enter the number of vertices: "))
    
    graph = {i: [] for i in range(n)}
    
    # Get edges from user
    for _ in range(n):
        u = int(input(f"Enter the start vertex for edge {_+1}: "))
        v = int(input(f"Enter the end vertex for edge {_+1}: "))
        if u in graph and v in graph:
            graph[u].append(v)
        else:
            print("Invalid vertex. Please enter a valid vertex number.")
            break
    
    # Measure execution t
    
    try:
        result = topological_sort(graph)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        print(f"Graph: {graph}")
        print(f"Topological Sort Order: {result}")
        print(f"Execution time: {execution_time:.6f} seconds")
    except ValueError as e:
        print(e)

# Calculate the average execution time (for a single measurement, it's just the time itself)
average_time = execution_times[0] if execution_times else 0

# Generate x-axis values (measurement)
measurements = list(range(1, num_measurements + 1))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(measurements, execution_times, marker='o', linestyle='-', color='dodgerblue', label='Execution Time')
plt.axhline(y=average_time, color='red', linestyle='--', label='Average Time')
plt.xlabel('Measurement')
plt.ylabel('Time (seconds)')
plt.title('Execution Time of Topological Sorting')
plt.legend()
plt.grid(True)
plt.show()
```
### Result : Thus, the python program to implement Topological Sorting using Decrease and Conquer has been executed successfully.

# I shall expeditiously conclude the outstanding updates.
## BY
# HELLO WORLD 
