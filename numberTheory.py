import collections
from collections import deque

"""
1. Modular Exponentiation
2. Modular multiplicative inverse
3. Primality Test | Set 2 (Fermat Method)
4. Euler’s Totient Function
5. Sieve of Eratosthenes
6. Convex Hull
7. Basic and Extended Euclidean algorithms
8. Segmented Sieve
9. Chinese remainder theorem
10. Lucas Theorem
"""


# 1. Modular Exponentiation: (x^y)%p
# O(logN) time and O(1) space
def power(x, y, p):
    # edge case
    x = x % p
    if x == 0:
        return 0

    res = 1
    while y > 0:
        if (y & 1) != 0:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res


# 2. Modular multiplicative inverse
def modInverse(a, m):
    # O(logM) w/ Euclidean Algorithm or Fermat's litter theorem
    """
    # Euclidean Algo
    a >= b,
    GCD(a, b) = GCD(max(a-b, b), min(a-b, b)) --> if a == b: return a
    GCD(a, b) = GCD(b, a % b) --> if b == 0: return a

    1) if a, b is coPrime,
    ax + by = GCD(a, b)
    here y = m, ax + bm = GCD(a, b)
                        = 1 (if a, b is coPrime)
                    ax  ~= 1

    # Fermat's litter theorem
    2) if m is prime number,
    """

    # naive O(M)
    """
    for n in range(1, m):
        if (a * n) % m == 1:
            return n
    """
    pass
    return


# 3. Primality Test | Set 2 (Fermat Method)
# prime test: naive solution: O(n**0.5) time and constant space
# Fermat Method: O(klogN) time
# pass: probabilistic method
def isPrime(n, k):
    pass
    return


# 4. Euler’s Totient Function
def phi(n):
    pass
    return


# 5. Sieve of Eratosthenes
# O(Nloglog(N)) time and O(N) space
def SieveOfEratosthenes(n):
    ans = []
    prime = [True for _ in range(n + 1)]
    for i in range(2, n + 1):
        if i ** i >= n:
            break
        if prime[i]:
            start = i
            for num in range(start * 2, n + 1, i):
                prime[num] = False

    for i in range(2, n + 1):
        if prime[i]:
            ans.append(i)
    return ans


# 6. Convex Hull
# Finding Convex Hull that is the smallest convex polygon that contains all the points of it.
def orientation(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y3 - y2) / (x3 - x2)
    return m1 - m2
    # m1-m2 same as checking (y2-y1)*(x3-x2) - (y3-y2)*(x2-x1)
    # positive: clockwise, neg: counterclockwise, 0: co-linear


def onSegment(p, q, r):
    if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
        return True
    return False


def areIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and onSegment(p1, q1, p2):
        return True
    if o2 == 0 and onSegment(p1, q1, q2):
        return True
    if o3 == 0 and onSegment(p2, q2, p1):
        return True
    if o4 == 0 and onSegment(p2, q2, q1):
        return True
    return False


def convexHull(points, n):
    # Jarvis’s Algorithm: O(n**2) time and O(n) space
    p = None
    valx, valy = float("inf"), -float("inf")
    for idx, (x, y) in enumerate(points):
        if valx > x:
            valx = x
            p = idx
        elif valx == x:
            if y > valy:
                valy = y
                p = idx
    start = p
    ans = []
    q = 0
    while True:
        ans.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
            if orientation(points[p], points[i], points[q]) == 0 and p != i and onSegment(points[p], points[q],
                                                                                          points[i]):
                q = i
        p = q
        if p == start:
            break
    return ans


# 7. Euclidean algorithms (Basic and Extended)
# O(log(min(a, b))) time and space
def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)


def gcdExtended(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = gcdExtended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


# 8. Segmented Sieve
# O(nloglogn) time and O(n**0.5) space
def segmentedSieve(n):
    limit = int(n**0.5)+1
    prime = SieveOfEratosthenes(limit)
    lo = limit
    hi = limit*2
    while lo < n:
        if hi > n:
            hi = n
        p = [True for _ in range(limit+1)]
        for num in prime:
            start = (lo // num) * num
            if start < lo:
                start += num
            for j in range(start, hi, num):
                p[j-lo] = False

        for j in range(limit+1):
            if p[j]:
                prime.append(lo+j)
        lo += limit
        hi += limit
    return prime


# 9. Chinese remainder theorem
# naive approach: O(MN) time and O(1) space
# better approach: O(NlogN) time and O(N) space
def chineseRemainderTheorem(nums, rems, mods):
    prod = 1
    n = len(nums)
    for n in mods:
        prod *= n

    # naive approach
    for num in range(1, prod + 1):
        for i in range(n):
            if num % nums[i] != rems[i]:
                break
            if i == n - 1:
                return num

    # modular inverse approach
    ans = 0
    for i in range(n):
        pp = prod / nums[i]
        x = gcdExtended(pp, nums[i])
        ans += pp * x * rems[i]
    return ans


# 10. Lucas Theorem: Compute nCr mod p
# DP approach: O(NR) time and O(R) space
# Lucas Theorem: O(P^2 log p N) and O(P) space
def combinationWithDP(n, r, m):
    memo = collections.defaultdict(int)

    def topdownDP(n, r):
        if (n, r) in memo:
            return memo[(n, r)]
        if n == r or r == 0:
            return 1

        memo[(n, r)] = topdownDP(n - 1, r - 1) + topdownDP(n - 1, r)
        return memo[(n, r)] % m

    def bottomUpDP(n, r):
        dp = [0 for _ in range(r + 1)]
        for i in range(1, n + 1):
            j = min(i, r)
            while j > 0:
                dp[j] = (dp[j] + dp[j - 1]) % m
                j -= 1
        return dp[r]

    return topdownDP(n, r)
    # return bottomUpDP(n, r)


def lucasTheorem(n, r, p):
    if r == 0:
        return 1
    ni, ri = n % p, r % p
    return combinationWithDP(ni, ri, p) * lucasTheorem(n // p, r // p, p)
