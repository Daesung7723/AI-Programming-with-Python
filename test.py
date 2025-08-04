import numpy as np

# Practice code: Various array creation functions

# 1. Create from a list
list1 = [1, 2, 3, 4, 5]
arr1 = np.array(list1)
print(f"np.array() result:\n{arr1}\n")

# 2. Create an array filled with 0s
# Create a float type array of shape (2, 3) by filling it with 0s
arr2 = np.zeros((2, 3)) # Corrected the typo here
print(f"np.zeros() result:\n{arr2}\n")

# 3. Create an array filled with 1s
# Create an int type array of shape (2, 3, 4) by filling it with 1s
arr3 = np.ones((2, 3, 4), dtype=np.int16)
print(f"np.ones() result:\n{arr3}\n")

# 4. Create an array with consecutive values
# An array that increases by 1 from 0 to 9
arr4 = np.arange(10)
print(f"np.arange(10) result:\n{arr4}\n")

# Create an array from 0 to 1 with an interval of 0.1
arr5 = np.arange(0, 1, 0.1)
print(f"np.arange(0, 1, 0.1) result:\n{arr5}\n")