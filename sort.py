"""
1. Binary Search
2. Search an element in a sorted and rotated array
3. Bubble Sort
4. Insertion Sort
5. Merge Sort
6. Heap Sort (Binary Heap)
7. Quick Sort
8. Interpolation Search
9. Find Kth Smallest/Largest Element In Unsorted Array
10. Given a sorted array and a number x, find the pair in array whose sum is closest to x
"""

# 1. Binary Search
def problem1(arr, target, boundary = (lo, hi)):
  if boundary is None:
    lo, hi = 0, len(arr)-1
  else:
    lo, hi = boundary
                
  while lo <= hi:
    mid = lo + (hi-lo)//2
    if arr[mid] == target:
      return mid
    elif arr[mid] > target:
      hi = mid-1
    else:
      lo = mid+1
  return -1

# 2. Search an element in a sorted and rotated array
# assumption: distinct values for O(logN) complexity!
def problem2TwoPass(arr, key):
  if len(arr) == 1:
    return 0 if arr[0] == key else -1
  
  def findPivot(arr):
    lo, hi = 0, len(arr)-1
    while lo < hi:
      mid = lo + (hi-lo)//2
      if arr[mid] > arr[hi]:
        lo = mid+1
      else:
        hi = mid
    return lo
  idx = findPivot(arr)
  if idx == 0:
    return problem1(arr, key, (idx, len(arr)-1))
  elif arr[-1] < key:
    return problem1(arr, key, (0, idx-1))
  else:
    return problem1(arr, key, (idx, len(arr)-1))

def problem2OnePass(arr, key):
  lo, hi = 0, len(arr)-1
  while lo <= hi:
    mid = lo + (hi-lo)//2
    if arr[mid] == target:
      return mid
            
    if arr[mid] > arr[hi]:
      if arr[lo] <= target < arr[mid]:
        hi = mid-1
      else:
        lo = mid+1
            
    else: # arr[mid] < arr[hi]
      if arr[mid] < target <= arr[hi]:
        lo = mid+1
      else:
        hi = mid-1
  return -1  
  
def problem2DupOnePass(arr, key):
  # one-pass solution
  lo, hi = 0, len(arr)-1
  while lo <= hi:
    mid = lo + (hi-lo)//2
    if arr[mid] == target:
      return mid
            
    if arr[mid] > arr[hi]:
      if arr[lo] <= target < arr[mid]:
        hi = mid-1
      else:
        lo = mid+1
            
    elif arr[mid] < arr[hi]: 
      if arr[mid] < target <= arr[hi]:
        lo = mid+1
      else:
        hi = mid-1
                    
    else:
      while hi >= mid and arr[hi] == arr[mid]:
        hi -= 1
  return -1  
  
# 3. Bubble Sort: O(n**2) time and O(1) space, Best case occurs when array is already sorted O(n) time.
def bubbleSort(arr):
  n = len(arr)
  for i in range(n):
    swap = False
    for j in range(n-i-1):
      if arr[i] > arr[i+1]:
        arr[i+1], arr[i] = arr[i], arr[i+1]
        swap = True
    if not swap:
      break
  return arr

# 4. Insertion Sort
def insertionSort(arr):
  n = len(arr)
  
  
  
  
  
  
  
  
  return arr

# 5. Merge Sort





