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
def problem1(arr, target, boundary=(lo, hi)):
    if boundary is None:
        lo, hi = 0, len(arr) - 1
    else:
        lo, hi = boundary

    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


# 2. Search an element in a sorted and rotated array
# assumption: distinct values for O(logN) complexity!
def problem2TwoPass(arr, key):
    if len(arr) == 1:
        return 0 if arr[0] == key else -1

    def findPivot(arr):
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if arr[mid] > arr[hi]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    idx = findPivot(arr)
    if idx == 0:
        return problem1(arr, key, (idx, len(arr) - 1))
    elif arr[-1] < key:
        return problem1(arr, key, (0, idx - 1))
    else:
        return problem1(arr, key, (idx, len(arr) - 1))


def problem2OnePass(arr, key, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid

        if arr[mid] > arr[hi]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        else:  # arr[mid] < arr[hi]
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def problem2DupOnePass(arr, key, target):
    # one-pass solution
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid

        if arr[mid] > arr[hi]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        elif arr[mid] < arr[hi]:
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

        else:
            while hi >= mid and arr[hi] == arr[mid]:
                hi -= 1
    return -1


# 3. Bubble Sort: O(n**2) time and O(1) space, Best case occurs when array is already sorted O(n) time.
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        swap = False
        for j in range(n - i - 1):
            if arr[i] > arr[i + 1]:
                arr[i + 1], arr[i] = arr[i], arr[i + 1]
                swap = True
        if not swap:
            break
    return arr


# 4. Insertion Sort: O(n**2) time in worst case, linear time when the array is already sorted, O(1) space, useful when
# binary insertion sort takes O(logn) time to insert the key but it results in O(n**2) time complexity due to the swap operation
def insertionSort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > arr[i]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# 5. Merge Sort: O(nlogn) time and O(n) space (for reconstructing a new array). For linked list, the merge operation requires O(1) space instead.
# binary insection sort takes O(n**2) time for swapping. Divide and conquer paradigm. --> Recursively split the array and merge the splits.
def mergeSort(arr):
    def merge(arr1, arr2):
        m, n = len(arr1), len(arr2)
        ans = []
        idx1, idx2 = 0, 0

        while idx1 < m and idx2 < n:
            if arr1[idx1] <= arr2[idx2]:
                ans.append(arr1[idx1])
                idx1 += 1
            else:
                ans.append(arr2[idx2])
                idx2 += 1

        if idx1 < m:
            ans.append(arr1[idx1])
            idx1 += 1

        if idx2 < n:
            ans.append(arr2[idx2])
            idx2 += 1
        return ans

    def splitAndMerge(arr):
        lo, hi = 0, len(arr) - 1
        if lo == hi:
            return arr

        mid = lo + (hi - lo) // 2
        l = splitAndMerge(arr[:mid + 1])
        r = splitAndMerge(arr[mid + 1:])
        return merge(l, r)

    return splitAndMerge(arr)


# 6. Heap Sort (Binary Heap): O(nlogn) time (like merge sort), O(1) space (like insection sort)!
# Heap sort algorithm has limited uses because Quicksort and Mergesort are better (faster) in practice.
def heapify(arr, n, i):  # max heap: O(logn) time
    l, r = 2 * i + 1, 2 * i + 2
    largest = i
    if l < n and arr[l] > largest:
        largest = l
    if r < n and arr[r] > largest:
        largest = r
    if i != largest:
        arr[i], arr[largest] = arr[i], arr[largest]
        heapify(arr, largest, i)


def heapSort(arr):
    n = len(arr)
    for i in range(n // 2 + 1, -1,
                   -1):  # total time complexity O(n)! # https://www.cs.bgu.ac.il/~ds122/wiki.files/Presentation09.pdf
        heapify(arr, n, i)

    for i in range(n - 1, 2, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 2)
    return arr


# 7. Quick Sort: in-place sorting (O(1) space) algorithm. O(nlogn) time on average, O(n**2) time in worst case
# worst case: O(n**2) time when the array is already sorted (either increasing or decreasing order)
def partition(start, end, arr):
    val = arr[end]  # One can randomly select a pivot for partitioning
    pos = start - 1
    for i in range(start, end):
        if arr[i] < val:
            pos += 1
            arr[pos], arr[i] = arr[pos], arr[i]
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    return i + 1


def quick_sort(start, end, arr):
    if start < end:
        idx = partition(start, end, arr)
        quick_sort(start, idx - 1, arr)
        quick_sort(idx + 1, end, arr)


# 8. Interpolation Search: it assums that the value of array is uniformly distributed
# pos  = lo + (hi-lo) * (k-arr[lo])/(arr[hi]-arr[lo])
def interpolationSearch(arr, lo, hi, target):
    while lo <= hi and arr[lo] <= target <= arr[hi] and arr[hi] != arr[lo]:
        pos = lo + (hi - lo) * (target - arr[lo]) // (arr[hi] - arr[lo])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            hi = pos - 1
        else:
            lo = pos + 1
    return lo if arr[lo] == target else -1


# 9. Find Kth Smallest/Largest Element In Unsorted Array: expected linear time!
# find kth smallest/largest:
# standard sort(O(nlogn)), using heap(O(nlogk) or O(n + klogn) for optimized case), quick select (O(n) on average, O(n**2) in worst case)
# Assumption for quickselect: all the values of elements are distinct.
def kthSmallest(arr, l, r, k):
    if k > 0 and k <= l - r + 1:
        # quickselect
        # use rand function for optimization
        piv = random.randint(l, r)
        arr[piv], arr[r] = arr[r], arr[piv]
        val = arr[r]
        i = l - 1
        for j in range(l, r):
            if arr[j] < val:
                i += 1
                arr[j], arr[i] = arr[i], arr[j]
        arr[i + 1], arr[r] = arr[r], arr[i + 1]

        if l + k - 1 == i + 1:
            return arr[i + 1]
        elif l + k - 1 < i + 1:
            return kthSmallest(arr, l, i, k)
        else:
            return kthSmallest(arr, i + 2, r, k - (i + 2 - l))


# 10. Given a sorted array and a number x, find the pair in array whose sum is closest to x
# Naive BF: O(n**2) time and constant space
# Sort and two-pointers: O(nlogn) time and constant space, O(n) if the array is already sorted
def findClosest(arr, n, x):  # similar to two Sum problem
    arr = sorted(arr)
    p1, p2 = 0, len(arr) - 1
    minDiff = float("inf")
    while p1 < p2:
        if minDiff > abs(arr[p1] + arr[p2] - x):
            minDiff = abs(arr[p1] + arr[p2] - x)
            ans = arr[p1], arr[p2]
        if arr[p1] + arr[p2] <= x:
            p1 += 1
        else:
            p2 -= 1
    return ans
