### 1A factorial 
```
import time
import matplotlib.pyplot as plt

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def measure_runtime(n):
    start_time = time.time()
    factorial(n)
    return time.time() - start_time

def main():
    ns = []
    runtimes = []
    results = []
    
    while True:
        try:
            n = int(input("Enter a number to compute factorial: "))
            if n < 0:
                print("Please enter a non-negative integer.")
                continue
            
            # Compute factorial and measure runtime
            start_time = time.time()
            fact_result = factorial(n)
            runtime = time.time() - start_time
            
            # Store values
            ns.append(n)
            runtimes.append(runtime)
            results.append(fact_result)
            
            # Print results
            print(f"Factorial of {n} is {fact_result}")
            print(f"Factorial computation runtime: {runtime:.4f}s")
            
            # Plotting the results
            plt.plot([0] + ns, [0] + runtimes, marker='o', linestyle='-', color='blue')
            plt.xlabel('Input Number (n)')
            plt.ylabel('Runtime (s)')
            plt.title('Recursive Factorial Runtime')
            plt.ylim(0, max(runtimes) * 1.1 if runtimes else 1)
            plt.grid(True)
            plt.show()
            
            # Ask if the user wants to continue
            if input("Continue? (yes/no): ").strip().lower() != 'yes':
                break
        
        except ValueError:
            print("Invalid input. Please enter an integer.")

if __name__ == "__main__":
    main()
```
##
### 1B sum of no's
```
import time
import matplotlib.pyplot as plt

def sum_of_numbers(n):
    return sum(range(n + 1))

def measure_runtime(n):
    start_time = time.time()
    sum_of_numbers(n)
    return time.time() - start_time

def main():
    ns = []
    runtimes = []
    results = []

    while True:
        try:
            n = int(input("Enter a number to compute the sum of numbers up to: "))
            if n < 0:
                print("Please enter a non-negative integer.")
                continue

            # Compute sum and measure runtime
            start_time = time.time()
            sum_result = sum_of_numbers(n)
            runtime = time.time() - start_time

            # Store values
            ns.append(n)
            runtimes.append(runtime)
            results.append(sum_result)

            # Print results
            print(f"Sum of numbers up to {n} is {sum_result}")
            print(f"Sum computation runtime: {runtime:.4f}s")

            # Plotting the results
            plt.plot([0] + ns, [0] + runtimes, marker='o', linestyle='-', color='blue')
            plt.xlabel('Input Number (n)')
            plt.ylabel('Runtime (s)')
            plt.title('Sum of Numbers Runtime')
            plt.ylim(0, max(runtimes) * 1.1 if runtimes else 1)
            plt.grid(True)
            plt.show()

            # Ask if the user wants to continue
            if input("Continue? (yes/no): ").strip().lower() != 'yes':
                break

        except ValueError:
            print("Invalid input. Please enter an integer.")

if __name__ == "__main__":
    main()
```
##

### 2A
##### AIM : To implement Strassen’s Matrix Multiplication using Divide and Conquer.
```
import numpy as np
import time

import matplotlib.pyplot as plt

def strassen(A, B):
    # Base case for recursion
    if len(A) == 1:
        return A * B

    # Splitting matrices into quadrants
    def split(matrix):
        row, col = matrix.shape
        r = row // 2
        c = col // 2
        return (matrix[:r, :c], matrix[:r, c:],
                matrix[r:, :c], matrix[r:, c:])

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    # Strassen's Algorithm
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

    # Combine submatrices into a single matrix
    def combine(C11, C12, C21, C22):
        top = np.hstack((C11, C12))
        bottom = np.hstack((C21, C22))
        return np.vstack((top, bottom))

    return combine(C11, C12, C21, C22)

def pad_matrix(matrix, size):
    """Pad matrix to the next power of 2"""
    r, c = matrix.shape
    new_size = 1
    while new_size < max(r, c):
        new_size *= 2
    if new_size > r:
        padded = np.zeros((new_size, c))
        padded[:r, :] = matrix
        matrix = padded
    if new_size > c:
        padded = np.zeros((new_size, new_size))
        padded[:, :c] = matrix
        matrix = padded
    return matrix

def measure_runtime(A, B):
    start_time = time.time()
    strassen(A, B)
    end_time = time.time()
    return end_time - start_time

def main():
    matrix_sizes = []
    runtimes = []

    while True:
        try:
            m = int(input("Enter number of rows for matrix A: "))
            n = int(input("Enter number of columns for matrix A (should match rows of matrix B): "))
            p = int(input("Enter number of columns for matrix B: "))
            
            if n != p:
                print("For multiplication, number of columns of matrix A must be equal to number of rows of matrix B.")
                continue

            print("Enter values for matrix A:")
            A = np.array([[float(input(f"Enter value for A[{i+1}][{j+1}]: ")) for j in range(n)] for i in range(m)])
            
            print("Enter values for matrix B:")
            B = np.array([[float(input(f"Enter value for B[{i+1}][{j+1}]: ")) for j in range(p)] for i in range(n)])
            
            # Padding matrices to next power of 2
            max_dim = max(m, n, p)
            size = 1
            while size < max_dim:
                size *= 2
            
            A_padded = pad_matrix(A, size)
            B_padded = pad_matrix(B, size)
            
            # Measure runtime
            runtime = measure_runtime(A_padded, B_padded)
            matrix_sizes.append(size)
            runtimes.append(runtime)
            print(f"Matrix multiplication runtime: {runtime:.4f} seconds")

            # Print results
            result = strassen(A_padded, B_padded)
            result = result[:m, :p]  # Un-pad the result matrix
            print("Result of matrix multiplication:")
            print(result)

            # Plotting the results
            plt.plot([0] + matrix_sizes, [0] + runtimes, marker='o', color='blue', linestyle='-', label='Runtime')
            plt.xlabel('Matrix Size (NxN)')
            plt.ylabel('Runtime (seconds)')
            plt.title('Strassen\'s Matrix Multiplication Runtime')
            plt.xscale('linear')
            plt.yscale('linear')
            plt.grid(True)
            
            # Set y-axis to start from 0 and go up to the max runtime plus a small buffer
            plt.ylim(0, max(runtimes) * 1.1 if runtimes else 1)
            plt.legend()
            plt.show()

            # Ask if the user wants to continue
            cont = input("Do you want to enter another set of matrices? (yes/no): ").strip().lower()
            if cont != 'yes':
                break

        except ValueError:
            print("Invalid input. Please enter numerical values.")

if __name__ == "__main__":
    main()
```
##
### 3 Decrease and Conquer - Topological Sorting
#### AIM : To implement Topological sorting using Decrease and Conquer.
#### Algorithm 
```
1. **Input:** Number of vertices, edges, and edge list.
2. **Topological Sort:**
   - Build graph and compute in-degrees.
   - Process vertices with zero in-degrees.
   - Check for cycles.
3. **Measure Runtime:** Record time before and after sorting.
4. **Plot Runtime:** Graph runtime versus number of vertices.
5. **Loop or Exit:** Ask user to continue or stop.
```
```
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict

def topological_sort(vertices, edges):
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        in_degree.setdefault(u, 0)
    queue = deque(v for v in in_degree if in_degree[v] == 0)
    sorted_list = []
    while queue:
        u = queue.popleft()
        sorted_list.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(sorted_list) == len(in_degree):
        return sorted_list
    raise ValueError("Graph is not a DAG (contains a cycle)")

def measure_runtime(vertices, edges):
    start_time = time.time()
    result = topological_sort(vertices, edges)
    return result, time.time() - start_time

def main():
    ns, runtimes = [], []
    while True:
        try:
            n = int(input("Enter the number of vertices: "))
            m = int(input("Enter the number of edges: "))
            if n <= 0 or m < 0:
                print("Invalid input. Try again.")
                continue
            edges = [tuple(map(int, input(f"Edge {_+1} (u v): ").split())) for _ in range(m)]
            vertices = set(u for edge in edges for u in edge) | set(range(n))
            result, runtime = measure_runtime(vertices, edges)
            ns.append(n)
            runtimes.append(runtime)
            print(f"Topological Sort Order: {result}")
            print(f"Runtime: {runtime:.4f}s")
            plt.plot([0] + ns, [0] + runtimes, marker='o', linestyle='-', color='blue')
            plt.xlabel('Number of Vertices')
            plt.ylabel('Runtime (s)')
            plt.title('Topological Sorting Runtime')
            plt.ylim(0, max(runtimes) * 1.1 if runtimes else 1)
            plt.grid(True)
            plt.show()
            if input("Continue? (yes/no): ").strip().lower() != 'yes':
                break
        except ValueError as e:
            print(f"Invalid input: {e}")

if __name__ == "__main__":
    main()
