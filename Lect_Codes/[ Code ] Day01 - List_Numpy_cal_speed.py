import time
import numpy as np
import matplotlib.pyplot as plt

# Define the different data sizes to test
data_sizes = [10**3, 10**4, 10**5, 10**6, 10**7]

# Lists to store the timing results
list_times = []
numpy_times = []

def test_list_performance(n):
    """Tests the performance of list operations."""
    a_list = list(range(n))
    b_list = list(range(n))
    
    start_time = time.time()
    result_list = [a_list[i]**2 + b_list[i]**2 for i in range(n)]
    end_time = time.time()
    
    return end_time - start_time, result_list

def test_numpy_performance(n):
    """Tests the performance of NumPy array operations."""
    a_np = np.arange(n, dtype=np.int64)
    b_np = np.arange(n, dtype=np.int64)
    
    start_time = time.time()
    result_np = a_np**2 + b_np**2
    end_time = time.time()
    
    return end_time - start_time, result_np

print("Starting performance comparison...")

# Loop through each data size
for n in data_sizes:
    print(f"Testing with {n:,} elements...")
    
    # --- Python List Test ---
    list_time, result_list = test_list_performance(n)
    list_times.append(list_time)
    
    # --- NumPy Array Test ---
    numpy_time, result_np = test_numpy_performance(n)
    numpy_times.append(numpy_time)
    
    # --- Result Verification ---
    # Verify that the results are the same
    if not np.allclose(result_list, result_np):
        print("Warning: Results are not the same!")

print("Comparison finished. Generating plot...")

# --- Plotting the Results ---
plt.figure(figsize=(10, 6))

plt.plot(data_sizes, list_times, marker='o', linestyle='-', label='Python List')
plt.plot(data_sizes, numpy_times, marker='o', linestyle='-', label='NumPy Array')

# Use a logarithmic scale for both axes for better visualization
plt.xscale('log')
plt.yscale('log')
plt.yscale('log')

plt.title('Performance Comparison: Python List vs. NumPy Array')
plt.xlabel('Number of Elements (Log Scale)')
plt.ylabel('Time Taken in Seconds (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()