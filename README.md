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
