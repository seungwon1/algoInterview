import sys
input = sys.stdin.readline
from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

def readList():
    return list(map(int, input().split()))
def readInt():
    return int(input())
def readInts():
    return map(int, input().split())
def readStr():
    return input().strip()

n, m = readInts()
topo = []
used = [0] * (2 * n)
comp = [-1] * (2 * n)
graph = [[] for _ in range(2 * n)]
graph_t = [[] for _ in range(2 * n)]
for _ in range(m):
    u, v = readInts()
    graph[u-1].append(v-1)
    graph_t[v-1].append(u-1)

# 2sat SCC - Kosaraju's algorithm
@bootstrap
def dfs1(node):
    used[node] = 1
    for to in graph[node]:
        if not used[to]:
            yield dfs1(to)
    topo.append(node)
    yield 1

@bootstrap
def dfs2(node, j):
    comp[node] = j
    for to in graph_t[node]:
        if comp[node] == -1:
            yield dfs2(to, j)
    yield 1

def find_scc():
    topo.clear()
    for i in range(2 * n):
        if not used[i]:
            dfs1(i)
    j = 0
    for i in range(2 * n):
        if comp[topo[2 * n - 1 - i]] == -1:
            dfs2(topo[2 * n - 1 - i], j)
            j += 1

    values = [0] * n
    for i in range(n):
        if comp[2*i] == comp[2*i + 1]:
            return 0
        values[i] = comp[2*i] > comp[2*i + 1]
    return 1, values

print(find_scc())
