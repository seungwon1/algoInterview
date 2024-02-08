import sys
input = sys.stdin.readline

n = 10**5
tin = [n+1] * n
low = [n+1] * n
vis = [0] * n
clk = [0]
scc = [[] for _ in range(n)]
stk = []
inStk = [0] * n
graph = [[] for _ in range(n)]
par = [-1] * n

def dfs(n):
    for i in range(n):
        if not vis[i]:
            dfsVisit_it(i)

def dfsVisit_it(node):
    st = [(node, 0)]
    while st:
        node, exp = st.pop()
        if exp:
            for to in graph[node]:
                if par[to] == node:
                    low[node] = min(low[node], low[to])
            if tin[node] == low[node]:
                while stk and stk[-1] != node:
                    scc[node].append(stk.pop())
                    inStk[scc[node][-1]] = 0
                scc[node].append(stk.pop())
                inStk[scc[node][-1]] = 0
            continue

        if vis[node]:
            continue

        st.append((node, 1))
        vis[node] = 1
        clk[0] += 1
        tin[node] = low[node] = clk[0]
        stk.append(node)
        inStk[node] = 1
        for to in graph[node]:
            if not vis[to]:
                st.append((to, 0))
                par[to] = node
            elif inStk[to]:
                low[node] = min(low[node], tin[to])


def dfsVisit_rc(node):
    clk[0] += 1
    tin[node], low[node] = clk[0], clk[0]
    vis[node] = 1
    stk.append(node)
    inStk[node] = 1
    for to in graph[node]:
        if not vis[to]:
            dfsVisit_rc(to)
            low[node] = min(low[node], low[to])
        elif inStk[to]:
            low[node] = min(low[node], tin[to])

    if tin[node] == low[node]:
        while stk and stk[-1] != node:
            scc[node].append(stk.pop())
            inStk[scc[node][-1]] = 0
        scc[node].append(stk.pop())
        inStk[scc[node][-1]] = 0



