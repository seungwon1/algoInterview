import sys

input = sys.stdin.readline


def readList():
    return list(map(int, input().split()))


def readInt():
    return int(input())


def readInts():
    return map(int, input().split())


def readStr():
    return input().strip()


# BF, EC, DB, CC, CL
def solve():
    return


for _ in range(int(input())):
    print(solve())

# 998244353


# return f"{a} + {mi + k - 1} = {a + mi + k - 1}"

# for recursion
sys.setrecursionlimit(100000)

# mod inverse a/b mod c
# modular expression of fraction ((a % m) * pow(b, m-2, m)) % m --> can be used for computing nCr % m !
# factorial and inverse factorial
"""
fac, invF = [1, 1], []
for i in range(2, n+1):
    fac.append((fac[-1]*i) % m)
invF.append(pow(fac[-1], m-2, m))
for i in range(n, 0, -1):
    invF.append(invF[-1] * i % m)
invF = invF[::-1]
"""

# hash collision
# https://codeforces.com/contest/1728/submission/171487693
from random import randint

RANDOM = randint(1, 10 ** 9)


class myHash(int):
    def __init__(self, x):
        int.__init__(x)

    def __hash__(self):
        return super(myHash, self).__hash__() ^ RANDOM


# rolling hash
def hash(arr, P, m):
    return [reduce(lambda h, x: (h * p + x) % m, arr) for p in P]


P = [113, 109]
m = [pow(10, 9) + 7, pow(10, 9) + 9, pow(10, 9) + 21, pow(10, 9) + 23]


# DSU
class DSU:
    def __init__(self, n):
        self.p = [i for i in range(n)]
        self.size = [1] * n

    def find(self, x):
        root = x
        while root != self.p[root]:
            root = self.p[root]

        while x != root:
            tmp = x
            x = self.p[x]
            self.p[tmp] = root
        return root

    def union(self, x, y):
        pa, pb = self.find(x), self.find(y)
        if pa == pb:
            return True
        if self.size[pa] < self.size[pb]:
            pa, pb = pb, pa
        self.p[pb] = pa
        self.size[pa] += self.size[pb]
        return False


# lazy segtree
# https://codeforces.com/contest/558/submission/216810021
# https://codeforces.com/contest/580/status/E
# https://codeforces.com/contest/580/submission/148154945
# https://codeforces.com/contest/580/submission/228666804
# https://codeforces.com/contest/1709/submission/165259913
# ope- min, min & cnt, gcd & lcm,
class SegTree:
    def __init__(self, n, e, ope, lst=[]):
        # closest N0 - power of 2 - s.t., 2*n - 1 < N0
        self.N0 = 2 ** (n - 1).bit_length()
        self.e = e
        self.ope = ope
        self.data = [e] * (2 * self.N0)
        if lst:
            for i in range(n):
                self.data[self.N0 + i] = lst[i]
            # self.build()
            for i in range(self.N0 - 1, 0, -1):
                self.data[i] = self.ope(self.data[2 * i], self.data[2 * i + 1])

    def build(self):
        for i in range(self.N0 - 1, 0, -1):
            self.data[i] = self.ope(self.data[2 * i], self.data[2 * i + 1])

    def update(self, i, x):
        i += self.N0
        self.data[i] = x
        while i > 1:
            i >>= 1
            self.data[i] = self.ope(self.data[2 * i], self.data[2 * i + 1])

    def add(self, i, x):
        self.update(i, x + self.get(i))

    def set(self, i, x):
        self.data[self.N0 + i] = x

    def query(self, l, r):
        if r <= l:
            return self.e
        lres = self.e
        rres = self.e
        l += self.N0
        r += self.N0
        while l < r:
            # odd --> add and shift to the right. o.w. (i.e., even) --> going upward ( >>= 1)
            if l & 1:
                lres = self.ope(lres, self.data[l])
                l += 1
            if r & 1:
                r -= 1
                rres = self.ope(self.data[r], rres)
            l >>= 1
            r >>= 1
        return self.ope(lres, rres)

    def get(self, i):
        return self.data[self.N0 + i]

    # customized iterative implementation - O(logN)
    # https://cp-algorithms.com/data_structures/segment_tree.html#searching-for-an-array-prefix-with-a-given-amount
    # Searching for the k-th element in the range
    # the same logic can be applied to the problems below: key -- convert into the prefix problems, modify k value
    # Searching for an array prefix with a given amount
    # Searching for the first element greater than a given amount
    def find_kth(self, l, r, k):
        k += self.query(0, l)
        if self.query(0, r) < k:
            return -1

        idx, f, b = 0, 1, len(self.data) - 1
        while f < b:
            if self.data[2 * idx] >= k:
                idx, f, b = 2 * idx, f, (f + b) // 2
            else:
                k -= self.data[2 * idx]
                idx, f, b = 2 * idx + 1, (f + b) // 2 + 1, b
        return idx

    # customized iterative implementation - O(logN)
    # binary search to find the first element on the right starting from the left
    def first_greater(self, l):
        return self.find_kth(0, n, self.query(0, l) + 1)

# Lazy Segment Tree

# https://codeforces.com/contest/580/status/E
# https://codeforces.com/contest/580/submission/148154945
# https://codeforces.com/contest/580/submission/228666804

# https://codeforces.com/contest/1709/submission/165259913

# https://codeforces.com/contest/558/problem/E
# https://codeforces.com/contest/558/submission/216810021
"""
        V: Initial sequence, leaf nodes
        OP: Merge operation between nodes/segments
        E: Identity element for nodes/segments. op(e, x) = op(x, e) = x
        Mapping: Apply operation F to segments
        COMPOSITION: Composition of F and G: returns F(G(seg))
        ID: Identity mapping: F(ID(seg)) = F(seg)
    """

class LazySegTree:
    def _update(self, k):
        self.d[k] = self.op(self.d[2 * k], self.d[2 * k + 1])

    def _all_apply(self, k, f):
        self.d[k] = self.mapping(f, self.d[k], self.length[k])
        if k < self.size:
            self.lz[k] = self.composite(f, self.lz[k])

    def _push(self, k):
        self._all_apply(2 * k, self.lz[k])
        self._all_apply(2 * k + 1, self.lz[k])
        self.lz[k] = self.identity

    # customized
    def mapping(self, a, x, l):
        if a == -1:
            return x
        if a == 0:
            return 0
        return l

    # customized
    def composite(self, a, b):
        return a if a >= 0 else b

    def __init__(self, V, OP, E):
        self.n = len(V)
        self.log = (self.n - 1).bit_length()
        self.size = 1 << self.log
        self.d = [E for i in range(2 * self.size)]
        self.length = [0] * (2 * self.size)
        self.e = E
        self.op = OP
        self.identity = -1
        self.lz = [-1] * self.size
        for i in range(self.n):
            self.d[self.size + i] = V[i]
            self.length[self.size + i] = 1
        for i in range(self.size - 1, 0, -1):
            self._update(i)
            self.length[i] = self.length[2 * i] + self.length[2 * i + 1]

    # assign x to point p w/ lazy prop
    def set(self, p, x):
        assert 0 <= p < self.n
        p += self.size
        for i in range(self.log, 0, -1):
            self._push(p >> i)
        self.d[p] = x
        for i in range(1, self.log + 1):
            self._update(p >> i)

    # query point p w/ lazy prop
    def get(self, p):
        assert 0 <= p < self.n
        p += self.size
        for i in range(self.log, 0, -1):
            self._push(p >> i)
        return self.d[p]

    # query range l, r w/ lazy prop
    def prod(self, l, r):
        assert 0 <= l <= r <= self.n
        if l == r:
            return self.e
        l += self.size
        r += self.size
        # lazy prop from the root
        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l:
                self._push(l >> i)
            if ((r >> i) << i) != r:
                self._push(r >> i)
        sml, smr = self.e, self.e
        while l < r:
            if l & 1:
                sml = self.op(sml, self.d[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.op(self.d[r], smr)
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    # query the whole array a[1...n]
    def all_prod(self):
        return self.d[1]

    # update point w/ lazy prop
    def apply_point(self, p, f):
        assert 0 <= p < self.n
        p += self.size
        for i in range(self.log, 0, -1):
            self._push(p >> i)
        self.d[p] = self.mapping(f, self.d[p], self.length[p])
        for i in range(1, self.log + 1):
            self._update(p >> i)

    # update query w/ lazy prop
    def apply(self, l, r, f):
        assert 0 <= l <= r <= self.n
        if l == r:
            return
        l += self.size
        r += self.size
        for i in range(self.log, 0, -1):
            if ((l >> i) << i) != l:
                self._push(l >> i)
            if ((r >> i) << i) != r:
                self._push((r - 1) >> i)
        l2, r2 = l, r
        while l < r:
            if l & 1:
                self._all_apply(l, f)
                l += 1
            if r & 1:
                r -= 1
                self._all_apply(r, f)
            l >>= 1
            r >>= 1
        l, r = l2, r2
        # update from the leaf
        for i in range(1, self.log + 1):
            if ((l >> i) << i) != l:
                self._update(l >> i)
            if ((r >> i) << i) != r:
                self._update((r - 1) >> i)

    # binary search starting from the l
    def max_right(self, l, g):
        assert 0 <= l <= self.n
        assert g(self.e)
        # null case
        if l == self.n:
            return self.n
        l += self.size
        for i in range(self.log, 0, -1):
            self._push(l >> i)
        sm = self.e
        while 1:
            while l % 2 == 0:
                l >>= 1
            # go left condition
            if not g(self.op(sm, self.d[l])):
                # loop up to the leaf node
                while l < self.size:
                    self._push(l)
                    l *= 2
                    if g(self.op(sm, self.d[l])):
                        sm = self.op(sm, self.d[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.d[l])
            l += 1
            # check l is power of 2
            if (l & -l) == l:
                break
        return self.n

    # binary search starting from the right
    def min_left(self, r, g):
        assert (0 <= r <= self.n)
        assert g(self.e)
        # null case
        if r == 0:
            return 0
        r += self.size
        for i in range(self.log, 0, -1):
            self._push((r - 1) >> i)
        sm = self.e
        while 1:
            r -= 1
            while r > 1 and (r % 2): r >>= 1
            # go left condition
            if not (g(self.op(self.d[r], sm))):
                # loop up to the leaf node
                while r < self.size:
                    self._push(r)
                    r = (2 * r + 1)
                    if g(self.op(self.d[r], sm)):
                        sm = self.op(self.d[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.d[r], sm)
            # check r is power of 2
            if (r & -r) == r:
                break
        return 0

class BIT:
    def __init__(self, n):
        self.bit = [0] * (n + 1)
        self.n = n

    def update(self, idx, val):
        idx += 1
        while idx <= self.n:
            self.bit[idx] += val
            idx += idx & (-idx)

    def prefixSum(self, idx):
        idx += 1
        ans = 0
        while idx > 0:
            ans += self.bit[idx]
            idx -= idx & (-idx)
        return ans

    def rangeSum(self, l, r):
        return self.prefixSum(r) - self.prefixSum(l - 1)


# concise template for segment tree
class SEG:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * 2 * self.n

    def query(self, l, r):
        l += self.n
        r += self.n
        ans = 0
        while l < r:
            if l & 1:
                ans = max(ans, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                ans = max(ans, self.tree[r])
            l >>= 1
            r >>= 1
        return ans

    def update(self, i, val):
        i += self.n
        self.tree[i] = val
        while i > 1:
            i >>= 1
            self.tree[i] = max(self.tree[i * 2], self.tree[i * 2 + 1])


class Solution:
    def lengthOfLIS(self, A: List[int], k: int) -> int:
        n, ans = max(A), 1
        seg = SEG(n)
        for a in A:
            a -= 1
            premax = seg.query(max(0, a - k), a)
            ans = max(ans, premax + 1)
            seg.update(a, premax + 1)
        return ans


# DFS iterative version (one pass) *** # topological sort
# https://codeforces.com/contest/1777/submission/196363442
def solve():
    n = readInt()
    m = 10 ** 9 + 7
    graph = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = readInts()
        graph[v].append(u)
        graph[u].append(v)

    st = [1]
    par = {1: -1}
    vis = [False] * (n + 1)
    d = [0] * (n + 1)
    while st:
        node = st[-1]
        if vis[node]:
            v = st.pop()
            if node == 1:
                continue
            d[par[v]] = max(d[par[v]], d[v] + 1)
            continue
        vis[node] = True
        for nei in graph[node]:
            if nei in par:
                continue
            par[nei] = node
            st.append(nei)
    return (2 ** (n - 1) * (sum(d) + n)) % m


# https://codeforces.com/contest/1746/submission/177436098
def solve():
    n, k = map(int, input().split())
    par = [0, 0] + list(map(int, input().split()))
    ind = [0] * (n + 1)
    score = [0] + list(map(int, input().split()))
    child = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        child[par[i]].append(i)
        ind[par[i]] += 1

    lo, hi = [0] * (n + 1), [0] * (n + 1)
    stack = [(1, 1, k)]
    while stack:
        node, bf, val = stack.pop()
        if bf:
            stack.append((node, 0, val))
            l = len(child[node])
            for ch in child[node]:
                stack.append((ch, bf, val // l))
        else:
            l = len(child[node])
            if not l:
                lo[node], hi[node] = val * score[node], (val + 1) * score[node]
            else:
                child[node].sort(key=lambda x: hi[x] - lo[x], reverse=True)
                diff = [hi[ch] - lo[ch] for ch in child[node]]
                v = sum([lo[ch] for ch in child[node]]) + sum(diff[:val % l])
                lo[node] = val * score[node] + v
                hi[node] = (val + 1) * score[node] + v + diff[val % l]
    return lo[1]


# LCA
# O(NlogN) precompute, O(logN) for each query


# 2-sat dfs
# https://codeforces.com/contest/1844/submission/214718987

# Inclusion - Exclusion Principle
# https://codeforces.com/blog/entry/64625
# finding v such that gcd(k, v) == 1 and 1 <= v <= m
def f(k, m):
    p = []
    for i in range(2, int(k ** 0.5 + 1)):
        if k % i == 0:
            p.append(i)
            while k % i == 0:
                k = k // i
    if k != 1:
        p.append(k)

    n = len(p)
    ans = m
    for i in range(1, 2 ** n):
        q = i
        cnt = 0
        a = 1
        for j in range(n):
            if q % 2 == 1:
                cnt += 1
                a *= p[j]
            q = q // 2
        ans += (-1) ** cnt * (m // a)
    return ans


# prime factorization
# 1) pollard_rho O(N^0.5 log N)
# https://codeforces.com/contest/1771/submission/184840112
from collections import Counter
from math import gcd


def memodict(f):
    """memoization decorator for a function taking a single argument"""

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


def pollard_rho(n):
    """returns a random factor of n"""
    if n & 1 == 0:
        return 2
    if n % 3 == 0:
        return 3

    s = ((n - 1) & (1 - n)).bit_length() - 1
    d = n >> s
    # for a in [2, 3, 5, 7]:
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        p = pow(a, d, n)
        if p == 1 or p == n - 1 or a % n == 0:
            continue
        for _ in range(s):
            prev = p
            p = (p * p) % n
            if p == 1:
                return gcd(prev - 1, n)
            if p == n - 1:
                break
        else:
            for i in range(2, n):
                x, y = i, (i * i + 1) % n
                f = gcd(abs(x - y), n)
                while f == 1:
                    x, y = (x * x + 1) % n, (y * y + 1) % n
                    y = (y * y + 1) % n
                    f = gcd(abs(x - y), n)
                if f != n:
                    return f
    return n


@memodict
def prime_factors(n):
    """returns a Counter of the prime factorization of n"""
    if n <= 1:
        return Counter()
    f = pollard_rho(n)
    return Counter([n]) if f == n else prime_factors(f) + prime_factors(n // f)


# prime factorization
# 2) Sieve O(NlogN) (pre-compute) + O(logN)
# https://codeforces.com/contest/1766/submission/187488325

# sieve: O(NlogN)
N = 10 ** 7 + 1
divPrimes = [-1] * N
isPrime = [True] * N
primes = []

for i in range(2, N):
    if isPrime[i]:
        primes.append(i)
        for j in range(i ** 2, N, i):
            divPrimes[j] = i
            isPrime[j] = False


# fast factorization: O(logN)
def fastFact(num, primes):
    p = []
    while primes[num] != -1:
        v = num
        d = primes[v]
        cnt = 0
        while v % d == 0:
            v //= d
            cnt += 1
        p.append((d, cnt))
        num = v
    if num > 1:
        p.append((num, 1))
    return p


# 3) Sieve O(NlogN) (pre-compute) + O(logN)

class SOE:
    def __init__(self, m):
        self.sieve = [-1] * (m + 1)
        self.prime = []
        for i in range(2, m + 1):
            if self.sieve[i] == -1:
                self.prime.append(i)
                self.sieve[i] = i
                j = 2 * i
                while j <= m:
                    self.sieve[j] = i
                    j += i

    def primes(self):
        # get primes
        return self.prime

    def fact(self, n):
        # prime factorization
        d = []
        while n != 1:
            p = self.sieve[n]
            e = 0
            while n % p == 0:
                e += 1
                n //= p
            d.append((p, e))
        return d

    def div(self, n):
        # get divisors
        c = [1]
        while n != 1:
            p = self.sieve[n]
            cnt = 1
            n //= p
            while self.sieve[n] == p:
                cnt += 1
                n //= p
            s = c.copy()
            for i in s:
                for j in range(1, cnt + 1):
                    c.append(i * (p ** j))
        return c


soe = SOE(10 ** 7)


# number of coprime pairs from 1 to N

def num_cp(n):
    N = 100005
    phi = [0] * N
    S = [0] * N
    for i in range(1, N):
        phi[i] = i
    for p in range(2, N):
        if phi[p] == p:
            phi[p] = p - 1
            for i in range(2 * p, N, p):
                phi[i] = (phi[i] // p) * (p - 1)
    for i in range(1, N):
        S[i] = S[i - 1] + phi[i]
    return S[i]


# Fast IO

import os
import sys
from io import BytesIO, IOBase

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")


# cycle detection
# outgoing edge = 1 for each node
# https://codeforces.com/contest/1867/submission/222953726
# outgoing edge > 1 for each node
# https://codeforces.com/contest/1217/submission/224730178 (위와 동일)

# cycle 있는 그래프에서 back edge 찾는법 dfs recursive 로 구현
# DFS template (recursive)
#
### iterative 로는 cycle back-edge 처리하는 것보다 구현 간단함.

# TLDR
# DFS template for cycle detection & back edge detection
# recursive https://codeforces.com/contest/1217/submission/207679394
class Graph:
    def __init__(self, n, m):
        self.nodes = n
        self.edges = m
        self.adj = [[] for i in range(n)]
        self.color = [1 for i in range(m)]
        self.vis = [0 for i in range(n)]
        self.colors = 1

    def add_edge(self, u, v, i):
        self.adj[u].append((i, v))

    def dfs(self, u):
        self.vis[u] = 1
        for i, v in self.adj[u]:
            if self.vis[v] == 1:
                self.colors = 2
                self.color[i] = 2
            if self.vis[v] == 0:
                self.dfs(v)
        self.vis[u] = 2

    def solve(self):
        for i in range(self.nodes):
            if self.vis[i] == 0:
                self.dfs(i)
        print(self.colors)
        print(' '.join(map(str, self.color)))


n, m = map(int, input().split(' '))
graph = Graph(n, m)
for i in range(m):
    u, v = map(int, input().split(' '))
    graph.add_edge(u - 1, v - 1, i)
graph.solve()


# iterative https://codeforces.com/contest/1217/submission/224738995
def solve():
    n, m = readInts()
    graph = [[] for _ in range(n)]
    exp, ans = [0] * n, [1] * m
    for i in range(m):
        u, v = readInts()
        graph[u - 1].append([i, v - 1])

    # iterative version for coloring back-edges
    for i in range(n):
        if exp[i] == 0:
            st = [(i, 0)]
            while st:
                node, marked = st.pop()
                if marked:
                    exp[node] = 2
                    continue
                else:
                    exp[node] = 1
                st.append((node, 1))
                for idx, nei in graph[node]:
                    if exp[nei] == 0:
                        st.append((nei, 0))
                    elif exp[nei] == 1:
                        ans[idx] = 2
    print(max(ans))
    return " ".join([str(v) for v in ans])


print(solve())


# dfs: finding an entry point (or a back-edge) in a cycle with N nodes and N edges
# iterative version
def dfs(b, graph):
    s = [b]
    vis, p = [0] * len(graph), [0] * len(graph)
    while s:
        c = s.pop()
        if vis[c]:
            goal = c
            break
        vis[c] = 1
        for j in graph[c]:
            if j != p[c]:
                p[j] = c
                s.append(j)
    return goal


from functools import reduce


# rolling hash
# base: 137 (or 113, 109) and mod pow(10, 9)+7 or pow(10, 9) + 9
# base > number of different values
# e.g. meta hacker-cup 2022 A2
# https://www.facebook.com/codingcompetitions/hacker-cup/2022/round-1/problems/A2/my-submissions
def check(a, b, idx, n):
    return all([a[(idx + i) % n] == b[i] for i in range(n)])


def hash(arr, P, m):
    return [reduce(lambda h, x: (h * p + x) % m, arr) for p in P]


def solve():
    n, k = readInts()
    a, b, m = readList(), readList(), pow(10, 9) + 7
    if k == 0:
        return "YES" if a == b else "NO"
    base = [pow(p, n, m) for p in [137]]
    pn = [pow(p, n, m) for p in base]
    ha, hb = hash(a, base, m), hash(b, base, m)
    for i in range(n):
        if ha == hb and check(a, b, i, n):
            if (n == 2 and k % 2 == i) or (n > 2 and ((k == 1 and i != 0) or k > 1)):
                return "YES"
        ha = [(ha[j] * base[j] + a[i] * (1 - pn[j])) % m for j in range(len(base))]
    return "NO"


for i in range(int(input())):
    print(f"Case #{i + 1}: {solve()}")

# LCA
# https://codeforces.com/contest/1878/submission/225472914
md = 998244353


class LCA:
    def __init__(self, parents, depth, aa):
        n = len(parents)
        self._depth = depth
        log = max(self._depth).bit_length()
        self._table = [parents] + [[-1] * n for _ in range(log)]
        self._bo = [aa] + [[0] * n for _ in range(log)]
        row0 = self._table[0]
        b0 = self._bo[0]
        for lv in range(log):
            row1 = self._table[lv + 1]
            b1 = self._bo[lv + 1]
            for u in range(n):
                if row0[u] == -1:
                    b1[u] = b0[u]
                else:
                    row1[u] = row0[row0[u]]
                    b1[u] = b0[u] | b0[row0[u]]
            row0, b0 = row1, b1

    def anc(self, u, v):
        diff = self._depth[u] - self._depth[v]
        if diff < 0: u, v = v, u
        diff = abs(diff)
        u, _ = self.up(u, diff)
        if u == v: return u
        for lv in range(self._depth[u].bit_length() - 1, -1, -1):
            anclv = self._table[lv]
            if anclv[u] != anclv[v]: u, v = anclv[u], anclv[v]
        return self._table[0][u]

    def up(self, u, dist):
        lv = 0
        b = 0
        while dist and u != -1:
            if dist & 1:
                b |= self._bo[lv][u]
                u = self._table[lv][u]
            lv, dist = lv + 1, dist >> 1
        return u, b

    def bisect(self, u, b):
        for lv in range(self._depth[u].bit_length())[::-1]:
            if self._bo[lv][u] | b == b: u = self._table[lv][u]
            if u == -1: return -1
        return u
