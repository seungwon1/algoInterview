import sys
sys.setrecursionlimit(5 * 10 ** 4)

# Strongly Connected Components: Tarjan's Algorithm
n = 2* (10**5)
graph = [[] for _ in range(n)]
scc = [[] for _ in range(n)]
clk = [0]
def findSCC(node, tin, low, st, ist):
    tin[node] = low[node] = clk[0]
    clk[0] += 1
    ist[node] = 1
    for to in graph[node]:
        if tin[to] == -1:
            findSCC(to, tin, low, st, ist)
        elif ist[to]:
            low[node] = min(low[node], tin[node])

    if tin[node] == low[node]:
        root = node
        while st and st[-1] != root:
            scc[root].append(st.pop())
            ist[scc[root][-1]] = 1
        scc[root].append(st.pop())
        ist[scc[root][-1]] = 1

def SCC():
    tin = [-1] * n
    low = [-1] * n
    st = []
    ist = [0] * n
    for i in range(n):
        if tin[i] == -1:
            findSCC(i, tin, low, st, ist)



