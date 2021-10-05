import collections
"----------------------------------------------------------------------------------------------------------------------"
"""
1. Longest Common Subsequence
2. Longest Increasing Subsequence
3. Edit Distance
4. Minimum Partition
5. Ways to Cover a Distance
6. Longest Path In Matrix
7. Subset Sum Problem
8. Optimal Strategy for a Game
9. 0-1 Knapsack Problem
10. Boolean Parenthesization Problem
"""
"----------------------------------------------------------------------------------------------------------------------"
"1. Longest Common Subsequence"
string1, string2 = "ABCDGH", "AEDFHR"
memo = collections.defaultdict(int)
# dp[i,j] LCS for substring s1[i:], s2[j:]
# can optimize space complexity by min(L(s1), L(s2))
# O(S1S2) time and space
# optimal solution lie in either 1) or 2), safe to consider 1 or 2
def LCS(i, j):
    if (i,j) in memo:
        return memo[(i,j)]
    if i >= len(string1) or j >= len(string2):
        return 0

    # 1)
    if string1[i] == string2[j]:
        memo[(i,j)] = 1 + LCS(i+1, j+1)
    # 2)
    else:
        memo[(i,j)] = max(LCS(i, j+1), LCS(i+1, j))
    return memo[(i,j)]


"----------------------------------------------------------------------------------------------------------------------"
"2. Longest Increasing Subsequence"
arr = [3,10,2,1,20]
memo = collections.defaultdict(int)
# DP[i]: LIS of substring arr[i:]
# O(N**2) time and O(N) space --> can be optimized by O(NlogN) w/ subsequence reconstruction using binary search
def LIS(idx):
    if idx in memo:
        return memo[idx]

    memo[idx] = 1
    for i in range(idx+1, len(arr)):
        if arr[idx] < arr[i]:
            memo[idx] = max(memo[idx], 1 + LIS(i))
    return memo[idx]

# One can convert LIS problem into a LCS by constructing a new sorted array from arr and finding LCS between the two arrays
arr1 = arr
arr2 = sorted(arr)
ans = LCS(0,0)

# O(NlogN) time solution: O(NlogN) time and O(N) space
# arr = [3,10,2,1,20]

def lisBinsearch(arr):
    stack = []
    for n in arr:
        if not stack or stack[-1] < n:
            stack.append(n)
            continue

        def findIdx(array, target):
            lo, hi = 0, len(array)-1
            ans = 0
            while hi >= lo:
                mid = lo + (hi-lo)//2
                if array[mid] > target:
                    hi = mid-1
                else:
                    ans = max(ans, mid)
                    lo = mid+1
            return ans
        idx = findIdx(arr, n)
        stack[idx] = n

    return len(stack)

# follow-up: find the all the combinations of LIS arrays
# 1. Design an algorithm to construct the longest increasing list. Also, model your solution using DAGs.
    # w/ DP, this can be done in O(n^2) time

# 2. Design an algorithm to construct all increasing lists of equal longest size.
    # w/ DP, this can be done in O(n^2) time

# 3. Is the above algorithm an online algorithm?
# 4. Design an algorithm to construct the longest decreasing list..


"----------------------------------------------------------------------------------------------------------------------"
"3. Edit Distance"
str1 = "geek", str2 = "gesek"
# O(S1 S2) time and space
# can optimize the space complexity by O(min(S1, S2)) via bottom-up DP
def editDitance(str1, str2):
    memo = collections.defaultdict(int)
    def TopDownDP(i, j):
        if (i,j) in memo:
            return memo[(i,j)]
        if i == len(str1) or j == len(str2):
            return max(len(str1)-i, len(str2)-j)

        if str1[i] == str2[j]:
            memo[(i,j)] = TopDownDP(i+1, j+1)
        else:
            memo[(i,j)] = min(TopDownDP(i+1, j), TopDownDP(i, j+1), TopDownDP(i+1, j+1)) + 1
        return memo[(i,j)]

    return TopDownDP(0,0)


"----------------------------------------------------------------------------------------------------------------------"
"4. Minimum Partition"
# extension of partition equal sum - O(NS) time and space where S is Sum(arr) // 2
# Minimum partition can be done in O(NS) time and O(S) space
arr = [1,6,11,5]
def minimumPartition(arr):
    S = sum(arr)
    midpoint = S // 2 + 1
    dp = [[False for _ in range(midpoint)] for _ in range(len(arr))]
    minDiff = float("inf")
    for i in range(len(arr)):
        for j in range(midpoint):
            if j == 0:
                dp[i][j] = True
            else:
                if i == 0:
                    if arr[i] == j:
                        dp[i][j] = True
                else:
                    if arr[i] <= j:
                        dp[i][j] = dp[i - 1][j - arr[i]] or dp[i - 1][j]
                    else:
                        dp[i][j] = dp[i - 1][j]

            if dp[i][j]:
                minDiff = min(minDiff, abs(j - (S - j)))
    return minDiff

"----------------------------------------------------------------------------------------------------------------------"
"5. Count number of ways to cover a distance"
# O(D) time and space (can optimize the space by constant)
dist = 7

def coverDistance(dist):
    dp = [0]*(dist+1) # dp[0]*3, dp = [1,2,2]
    dp[0] = 1
    for i in range(3, len(dp)):
        dp[i] = (dp[i - 1] + dp[i - 2] + dp[i - 3])
        """
        if i+1 < len(dp):
            dp[i+1] += dp[i]
        if i+1 < len(dp):
            dp[i+2] += dp[i]
        if i+1 < len(dp):
            dp[i+3] += dp[i]
        """
        # for i in range(3, n + 1):
        #    count[i] = (count[i-1] + count[i-2] + count[i-3]) or
                                        # ways[i % 3] = ways[(i - 1) % 3] + ways[(i - 2) % 3] + ways[(i - 3) % 3]
    return dp[-1]

"----------------------------------------------------------------------------------------------------------------------"
"6. Find the longest path in a matrix with given constraints"
mat = [[1,2,9], [5,3,8], [4,6,7]]

#DFS like approach!
# O(mn) time and space
def longestPathMatrix(mat):
    m, n = len(mat), len(mat[0])
    memo = collections.defaultdict(int)

    def dfs(pos):
        if pos in memo:
            return memo[pos]

        val = 0
        for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = pos[0] + move[0], pos[1] + move[1]
            if nx >= 0 and ny >= 0 and nx < m and ny < n:
                if mat[nx][ny] < mat[pos[0]][pos[1]]:
                    val = max(val, dfs((nx, ny)))
        memo[pos] = val + 1
        return memo[pos]

    ans = 1
    for i in range(m):
        for j in range(n):
            ans = max(ans, dfs((i,j)))
    return ans
"----------------------------------------------------------------------------------------------------------------------"
"7. Subset Sum Problem --> same as 4. Minimum Partition"
"""
NP-Complete (There is no known polynomial time solution for this problem): nondeterministic polynomial-time complete
Thus, we solve the problem in Pseudo-polynomial time use the Dynamic programming which is O(SN) time

arr = [3,34,4,12,5,2]
s = 9
"""
"----------------------------------------------------------------------------------------------------------------------"
"8. Optimal Strategy for a Game"
arr = [8,15,3,7]
# O(n**2) time and O(n) space
def optimalSTforGame(arr):
    """
    memo = collections.defaultdict(int)
    def TopdownDP(i, j, player):
        if i > j:
            return 0
        if (i,j) in memo:
            return memo[(i,j)]

        memo[(i, j)] = max(-TopdownDP(i + 1, j, not player) + arr[i], -TopdownDP(i, j - 1, not player) + arr[j])
        return memo[(i,j)]

    return TopdownDP(0, len(arr)-1, True) >= 0
    """
    dp = [[0 for _ in range(len(arr))] for _ in range(len(arr))]
    for i in range(len(arr) - 1, -1, -1):
        for j in range(i, len(arr)):
            if i == j:
                dp[i][j] = arr[i]
            else:
                dp[i][j] = max(arr[j] - dp[i][j - 1], arr[i] - dp[i + 1][j])
    return dp[0][len(arr) - 1] >= 0
"----------------------------------------------------------------------------------------------------------------------"
"9. 0-1 Knapsack Problem"
memo = collections.defaultdict(int)
# array value, weight
value, weight = [60, 100, 120], [10, 20, 30]
W = 50
# 0-1 knapsack
def knapsack(idx, W):
    if idx in memo:
        return memo[idx]
    if idx == len(value) or W == 0:
        return 0

    if weight[idx] > W:
        memo[idx] = knapsack(idx+1, W)
    else:
        memo[idx] = max(value[idx] + knapsack(idx+1, W-weight[idx]), knapsack(idx+1, W))
    return memo[idx]

"----------------------------------------------------------------------------------------------------------------------"
"10. Boolean Parenthesization Problem"
sym, ops = ["T", "F", "T", "T"], ["^", "&", "|"]
# the way that I initially implemented is correct - a bit verbose implementation
# below code adapted from GFG
# O(n**2) time and space

def parenthesis_count(Str, i, j, isTrue, dp):
    """
    memo = collections.defaultdict(int)
    def TopDownDP(i, j, boolean):

        if i == j:
            return 1 if boolean == sym[i] else 0

        if i + 1 == j:
            if ops[i] == "^":
                res = sym[i] ^ sym[j]
            elif ops[i] == "&":
                res = sym[i] and sym[j]
            else:
                res = sym[i] or sym[j]
            return 1 if res == boolean else 0

        memo[(i,j, boolean)] = 1

        for k in range(i, j):
        """
    if (i > j):
        return 0
    if (i == j):
        if (isTrue == 1):
            return 1 if Str[i] == 'T' else 0
        else:
            return 1 if Str[i] == 'F' else 0

    if (dp[i][j][isTrue] != -1):
        return dp[i][j][isTrue]
    temp_ans = 0
    for k in range(i + 1, j, 2):
        if (dp[i][k - 1][1] != -1):
            leftTrue = dp[i][k - 1][1]
        else:
            # Count number of True in left Partition
            leftTrue = parenthesis_count(Str, i, k - 1, 1, dp)

        if (dp[i][k - 1][0] != -1):
            leftFalse = dp[i][k - 1][0]
        else:
            # Count number of False in left Partition
            leftFalse = parenthesis_count(Str, i, k - 1, 0, dp)

        if (dp[k + 1][j][1] != -1):
            rightTrue = dp[k + 1][j][1]
        else:
            # Count number of True in right Partition
            rightTrue = parenthesis_count(Str, k + 1, j, 1, dp)

        if (dp[k + 1][j][0] != -1):
            rightFalse = dp[k + 1][j][0]
        else:
            # Count number of False in right Partition
            rightFalse = parenthesis_count(Str, k + 1, j, 0, dp)

        # Evaluate AND operation
        if (Str[k] == '&'):
            if (isTrue == 1):
                temp_ans = temp_ans + leftTrue * rightTrue
            else:
                temp_ans = temp_ans + leftTrue * rightFalse + leftFalse * rightTrue + leftFalse * rightFalse
        # Evaluate OR operation
        elif (Str[k] == '|'):
            if (isTrue == 1):
                temp_ans = temp_ans + leftTrue * rightTrue + leftTrue * rightFalse + leftFalse * rightTrue
            else:
                temp_ans = temp_ans + leftFalse * rightFalse

        # Evaluate XOR operation
        elif (Str[k] == '^'):
            if (isTrue == 1):
                temp_ans = temp_ans + leftTrue * rightFalse + leftFalse * rightTrue
            else:
                temp_ans = temp_ans + leftTrue * rightTrue + leftFalse * rightFalse
        dp[i][j][isTrue] = temp_ans
    return temp_ans

def countWays(N, S):
    dp = [[[-1 for k in range(2)] for i in range(N + 1)] for j in range(N + 1)]
    return parenthesis_count(S, 0, N - 1, 1, dp)
"----------------------------------------------------------------------------------------------------------------------"












