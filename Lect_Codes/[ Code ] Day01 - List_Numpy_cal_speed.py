import time
import numpy as np
import matplotlib.pyplot as plt

# Define the different data sizes to test
data_sizes = [10**3, 10**4, 10**5, 10**6, 10**7]

# Lists to store the timing results
list_times = []
numpy_times = []

print("Starting performance comparison...")

# Loop through each data size
for n in data_sizes:
    print(f"Testing with {n:,} elements...")
    
    # --- Python List Test ---
    a_list = list(range(n))
    b_list = list(range(n))
    result_list = [0] * n
    
    start_time_list = time.time()
    for i in range(n):
        result_list[i] = a_list[i]**2 + b_list[i]**2
    end_time_list = time.time()
    list_times.append(end_time_list - start_time_list)
    
    # --- NumPy Array Test ---
    a_np = np.arange(n, dtype=np.int64) # Use int64 to prevent overflow on large numbers
    b_np = np.arange(n, dtype=np.int64)
    
    start_time_np = time.time()
    result_np = a_np**2 + b_np**2
    end_time_np = time.time()
    numpy_times.append(end_time_np - start_time_np)

print("Comparison finished. Generating plot...")

# --- Plotting the Results ---
plt.figure(figsize=(10, 6))

plt.plot(data_sizes, list_times, marker='o', label='Python List')
plt.plot(data_sizes, numpy_times, marker='o', label='NumPy Array')

# Use a logarithmic scale for both axes for better visualization
plt.xscale('log')
plt.yscale('log')

plt.title('Performance Comparison: Python List vs. NumPy Array')
plt.xlabel('Number of Elements (Log Scale)')
plt.ylabel('Time Taken in Seconds (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()