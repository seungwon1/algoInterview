import collections

"""
1. Reverse an array without affecting special characters
2. All Possible Palindromic Partitions
3. Count triplets with sum smaller than a given value
4. Convert array into Zig-Zag fashion
5. Generate all possible sorted arrays from alternate elements of two given sorted arrays
6. Pythagorean Triplet in an array
7. Length of the largest subarray with contiguous elements
8. Find the smallest positive integer value that cannot be represented as sum of any subset of a given array
9. Smallest subarray with sum greater than a given value
10. Stock Buy Sell to Maximize Profit
"""


# 1. Reverse an array without affecting special characters
# O(n) time and O(n) space
def reverseSting(text):
    n = len(text)
    s = list(text)
    front, back = 0, n - 1
    while front < back:
        if not s[front].isalpha():
            front += 1
            continue

        if not s[back].isalpha():
            back -= 1
            continue

        s[front], s[back] = s[back], s[front]
        front += 1
        back -= 1

    return "".join(s)


# 2. All Possible Palindromic Partitions
# O(2**N N) time and O(N**2) space
def allPalPartitions(string: str):
    def savePalindrome(start, end, s, arr):
        n = len(s)
        while start >= 0 and end < n and s[start] == s[end]:
            arr[start].append(end)
            start -= 1
            end += 1
        return arr

    def backtrack(idx, tmp, ans, s, arr):
        if idx == len(s):
            ans.append(list(tmp))
            return

        for nextIdx in arr[idx]:
            tmp.append(s[idx:nextIdx + 1])
            backtrack(nextIdx + 1, tmp, ans, s, arr)
            tmp.pop()
        return ans

    n = len(string)
    pd = collections.defaultdict(list)
    for i, c in enumerate(string):
        pd = savePalindrome(i, i, string, pd)
        pd = savePalindrome(i, i + 1, string, pd)

    return backtrack(0, [], [], string, pd)


# 3. Count triplets with sum smaller than a given value
# BF: O(n**3) time and constant space
# better approach: O(n**2) time
def countTriplets(arr, n, s):
    arr = sorted(arr)
    ans = 0
    for i in range(n):
        targetVal = s - arr[i]
        front, back = i + 1, n - 1
        while front < back:
            if arr[front] + arr[back] < targetVal:
                ans += back - front
                front += 1
            else:
                back -= 1
    return ans


# 4. Convert array into Zig-Zag fashion
# O(N) time and O(1) space
def zigZag(arr, n):
    n = len(arr)
    for i in range(n - 1):
        if i % 2 == 0:
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        else:
            if arr[i] < arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return


# 5. Generate all possible sorted arrays from alternate elements of two given sorted arrays
# backtracking: O((M+N)2^(M+N)) time and O(2^(M+N)) space
# better approach: O(N**2) time and O(M+N) space w/ recursion ??
def generate(A, B, m, n):
    def mergeList(A, B):
        i, j, m, n = 0, 0, len(A), len(B)
        AB = []
        while i < m and j < n:
            if A[i] <= B[j]:
                AB.append((A[i], 0))
                i += 1
            else:
                AB.append((B[j], 1))
                j += 1
        while i < m:
            AB.append((A[i], 0))
            i += 1
        while j < n:
            AB.append((B[j], 1))
            j += 1
        return AB

    AB = mergeList(A, B)
    sortedArr = set()

    def backtrack(idx, tmp):
        v = tuple(tmp)
        if v in sortedArr:
            return

        if len(tmp) % 2 == 0 and tmp:
            print(tmp)
            sortedArr.add(v)

        for i in range(idx, m + n):
            if len(tmp) % 2 == 0:
                if AB[i][1] == 0 and (not tmp or AB[i] > tmp[-1]):
                    tmp.append(AB[i][0])
                    backtrack(i + 1, tmp)
                    tmp.pop()
            else:
                if AB[i][1] == 1 and AB[i] > tmp[-1]:
                    tmp.append(AB[i][0])
                    backtrack(i + 1, tmp)
                    tmp.pop()
        return

    backtrack(0, [])
    return sortedArr


# 6. Pythagorean Triplet in an array
# O(N**2) time and O(N) space
def isTriplet(ar, n):
    hashtable = {}
    for num in ar:
        hashtable[num ** 2] = True
    for i in range(n):
        for j in range(i + 1, n):
            if ar[i] ** 2 + ar[j] ** 2 in hashtable:
                return True
    return False


# 7. Length of the largest subarray with contiguous elements
# Assumption: array contains distinct integers
# BF: O(N**3) time and O(N**2) space
# O(N**2) time and constant space
def findLength(arr, n):
    ans = 1 if n > 0 else 0
    for i in range(n):
        minValue = arr[i]
        arraySum = arr[i]
        for j in range(i + 1, n):
            minValue = min(minValue, arr[j])
            arraySum += arr[j]
            if arraySum - minValue * (j - i + 1) == (j - i) * (j - i + 1) // 2:
                ans = max(ans, j - i + 1)
    return ans


# 7-2. follow-up: what if array contains duplicates values? --> use counter to ensure that all the elements are distinct
def findLengthWithDuplicates(arr, n):
    ans = 1 if n > 0 else 0
    for i in range(n):
        minValue = arr[i]
        maxValue = arr[i]
        tmp = set()
        tmp.add(arr[i])
        for j in range(i + 1, n):
            if arr[j] in tmp:
                break
            minValue = min(minValue, arr[j])
            maxValue = max(maxValue, arr[j])
            if maxValue - minValue == j - i:
                ans = max(ans, j - i + 1)
    return ans


# 8. Find the smallest positive integer value that cannot be represented as sum of any subset of a given array
# BF: O(2**N) time and O(2**N) space
# better approach: O(N) time and O(1) space **
def findSmallest(arr, n):
    smallestPosVal = 1
    for i in range(n):
        if arr[i] > smallestPosVal:
            break
        smallestPosVal += arr[i]
    return smallestPosVal


# 9. Smallest subarray with sum greater than a given value
# brute force: O(n**2) time and O(1) space
# better approach: O(N) time and constant space
def smallestSubWithSum(arr, n, x):
    if x <= 0:
        return []
    p1, p2 = 0, 0
    curr = 0
    minLength = float("inf")
    while p2 < n:
        while curr <= x and p2 < n:
            curr += arr[p2]
            p2 += 1

        while curr > x and p1 < p2:
            minLength = min(p2 - p1, minLength)
            curr -= arr[p1]
            p1 += 1

    return minLength


# 9-2. follow-up
# 1) # of subarray cums up to the given value - O(N) time and O(N) space w/ hashmap
# 2) find the subarray that sums up to the given value - O(N) time and O(1) space
# BF: O(n**2) time and O(n) space using prefix sum and iterate over all (i,j) pairs
# better solution: linear time and constant space

# 9-2-1
def subArraySumHashmap(arr, n, s):
    prefixSum = collections.defaultdict(set)
    curr = 0
    ans = 0
    for i, v in enumerate(arr):
        curr += v
        if curr == s:
            ans += 1
        if curr - s in prefixSum:
            ans += len(prefixSum[curr - s])
        prefixSum[curr].add(i)
    return ans


# 9-2-2 ***
# convert negative values into positive values and do linear scan w/ sliding window technique
def subArraySum(arr, n, s):
    minValue = abs(min(arr))
    arr = [e + minValue for e in arr]
    p1 = 0
    curr = arr[0]
    for i in range(1, n+1):
        while curr > (i - p1) * minValue + s and p1 < i:
            curr -= arr[p1]
            p1 += 1

        if curr - (i - p1) * minValue == s:
            return p1, i

        if i < n:
            curr += arr[i]

    return -1



# 10. Stock Buy Sell to Maximize Profit
# multiple transactions: O(N) time and O(1) space
def stockBuySell(price, n):
    ans = 0
    for i in range(1, n):
        if price[i] - price[i - 1] > 0:
            ans += price[i] - price[i - 1]
    return ans


# a pair of transaction: O(N) time and O(1) space
def stockBuySellOneShot(prices, n):
    prev = prices[0]
    ans = 0
    for i in range(1, n):
        if prices[i] >= prev:
            ans = max(ans, prices[i] - prev)
        else:
            prev = prices[i]
    return ans
