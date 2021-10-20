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
def problem1(arr, target):
  lo, hi = 0, len(arr)-1
  mid = lo + (hi-lo)//2
  if arr[mid] == target:
    return mid
  elif arr[mid] > target:
    hi = mid-1
  else:
    lo = mid+1
  return -1

# 2. Search an element in a sorted and rotated array
def problem2(arr, n, key):
  return
