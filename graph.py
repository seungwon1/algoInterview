import collections
import heapq
from collections import deque

"----------------------------------------------------------------------------------------------------------------------"
"""
1. Breadth First Search (BFS)
2. Depth First Search (DFS)
3. Shortest Path from source to all vertices **Dijkstra**
4. Shortest Path from every vertex to every other vertex **Floyd Warshall**
    + Bellman-Ford
5. To detect cycle in a Graph **Union Find**
6. Minimum Spanning tree **Prim**
7. Minimum Spanning tree **Kruskal**
8. Topological Sort
9. Boggle (Find all possible words in a board of characters), Boggle | Set 2 (Using Trie)
    212. Word Search II
10. Bridges in a Graph, Articulation Points (or Cut Vertices) in a Graph (Tarjan's algorithm)
    1192. Critical Connections in a Network
        
"""
"----------------------------------------------------------------------------------------------------------------------"
"1. Breadth First Search (BFS)"


def BFS(graph, root):
    dq = deque([root])
    visited = set()
    visited.add(root)
    while dq:
        node = dq.popleft()
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei)
                dq.append(nei)
                nei.depth = node.depth + 1
    return


"----------------------------------------------------------------------------------------------------------------------"
"2. Depth First Search (DFS)"  ####
parent = {}


def DFS(graph):
    parent = {}  # parent pointer!
    for node in graph:
        if node not in parent:
            parent[node] = None
            DFS_visit(graph, node)


def DFS_visit(graph, node):
    for nei in graph[node]:
        if nei not in parent:
            node[nei] = node
            DFS_visit(graph, nei)


"----------------------------------------------------------------------------------------------------------------------"
"3. Shortest Path from source to all vertices **Dijkstra**"


# In terms of shortest path, BFS can be used only in uniform weight whereas Dijkstra can be used in non-negative weights
def dijkstra(graph, src, dst):
    values = [float("inf")]
    values[src] = 0
    heap = []
    heapq.heappush((0, src))

    while heap:
        cost, node = heapq.heappop(heap)
        if node == dst:
            return cost
        for nei, ac in graph[node]:
            if values[nei] > cost + ac:
                values[nei] = cost + ac
                heapq.heappush(heap, (values[nei], nei))
    return -1


# O(ElogE) time and O(E+V) space
"----------------------------------------------------------------------------------------------------------------------"
"4. Shortest Path from every vertex to every other vertex **Floyd Warshall + Bellman Ford"


def floydWarshall(graph, nodes):  # shortest distance for any (i,j) pair ####
    V = len(nodes)
    values = [[[float("inf") for _ in range(V)] for _ in range(V)] for _ in range(V)]

    for i in nodes:
        for j in nodes:
            for k in nodes:
                values[i][k] = min(values[i][k], graph[i][j] + graph[j][k])
    return


# O(V^3) time and space

def bellmanFord(graph, src, dst, edges):  ####
    # shortest distance for a single st and dst. can be used in negative weight graph
    V = len(graph)  # number of vertices
    pred = {}
    values = [float("inf")] * V
    values[src] = 0
    for i in range(V):
        for (u, v) in edges:
            if values[v] > values[u] + graph[u][v]:
                values[v] = values[u] + graph[u][v]
                pred[v] = u

    # detect negative cycle
    for (u, v) in edges:
        if values[v] > values[u] + graph[u][v]:
            return False

    return True


# O(V^2) time and O(V) space
"----------------------------------------------------------------------------------------------------------------------"
"5. To detect cycle in a Graph **Union Find**"
# can be used to group elements into different disjoint sets. better than naive DFS or BFS in some problems
l = 10


class DSU(object):
    def __init__(self):
        self.p = range(l)
        self.size = [1] * (l)

    def find(self, x):  # can optimize this by path compression
        if x not in self.p:
            self.p[x] = x

        root = x
        while root != self.p[root]:
            root = self.p[root]
            if root not in self.p:
                self.p[root] = root

        while x != root:
            oldroot = self.p[x]
            self.p[x] = root
            x = oldroot

        return root

    def union(self, x, y):  # can optimize this w/ union by rank
        pa, pb = self.find(x), self.find(y)
        if pa == pb:
            return False  # detect a cycle

        if self.size[pa] >= self.size[pb]:
            self.size[pa] += self.size[pb]
            self.p[pb] = pa

        else:
            self.size[pb] += self.size[pa]
            self.p[pa] = pb
        return True


# O(N alpha(N)) time and O(E+V) space where alpha(N) is inverse ackermann which is similar to logN
"----------------------------------------------------------------------------------------------------------------------"
"6. Minimum Spanning tree **Prim**: connect element with minimal weights"


def mstPrim(graph, edges, root):  # Dijkstra
    pred = {}
    values = [float("inf")] * len(graph)
    values[root] = 0
    heap = []
    heapq.heappush(heap, (0, root))
    S = set()
    cost = 0
    while heap:
        weight, node = heapq.heappop(heap)
        if node in S:
            continue
        S.add(node)
        cost += weight
        for nei, w in graph[node]:
            if w not in S and values[nei] > w:
                values[nei] = w
                pred[nei] = node  ##
                heapq.heappush(heap, (w, nei))

    return root, cost


# O(ElogE) time and O(E+V) space
"----------------------------------------------------------------------------------------------------------------------"
"7. Minimum Spanning tree **Kruskal**: connect element with minimal weights"


def mstKruskal(graph, edges):  # union-find
    dsu = DSU()
    cost = 0
    A = {}
    for w, u, v in sorted(edges):
        pu, pv = dsu.find(u), dsu.find(v)
        if pu == pv:
            continue
        dsu.union(u, v)
        A.add(u)
        A.add(v)
        cost += w
    return A, cost


# O(ElogE) time and O(E+V) space
"----------------------------------------------------------------------------------------------------------------------"
"8. Topological Sort: ordering elements in directly acyclic graph, application: job (course) scheduling / ordering"


def topologicalSortBFS(graph, edges):
    indegree = collections.defaultdict(int)
    for (u, v) in edges:  # u to v direction
        indegree[v] += 1

    noindegree = []
    for (u, v) in edges:
        if u not in indegree:
            noindegree.append(u)

    ans = []
    while noindegree:
        node = noindegree.pop()
        ans.append(node)
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                noindegree.append(nei)
    return ans if len(ans) == len(graph) else []


def topologicalSortDFS(graph):
    stack = []
    parent = {}

    def tpDFS(node):
        for nei in graph[node]:
            if nei not in parent:
                parent[nei] = node
                tpDFS(nei)
        stack.append(node)
        return

    for v in graph:
        if v not in parent:
            parent[v] = None
            tpDFS(v)
    return stack[::-1]


"----------------------------------------------------------------------------------------------------------------------"
"9. Boggle (Find all possible words in a board of characters), Boggle | Set 2 (Using Trie)"

"----------------------------------------------------------------------------------------------------------------------"
"10. Bridges in a Graph, Articulation Points (or Cut Vertices) in a Graph (Tarjan's algorithm)"
"10-1 articulation points"
"10-2 Bridge in a Graph"


class solution10(object):

    def __init__(self, n):
        self.n = n
        self.T = 0
        self.dis, self.low = [float("inf")] * n, [float("inf")] * n
        self.AP = [False] * n
        self.parents = {}
        self.visited = set()

    def findAP(self, graph, node):

        self.visited.add(node)
        self.dis[node], self.low[node] = self.T, self.T
        self.T += 1
        child = 0
        for nei in graph[node]:
            if nei not in self.visited:
                child += 1
                self.parents[nei] = node
                self.findAP(graph, nei)
                self.low[node] = min(self.low[node], self.low[nei])

                if node not in self.parents and child > 1:
                    self.AP[node] = True
                if node in self.parents and self.low[nei] >= self.low[node]:
                    self.AP[node] = True

            elif self.parents[nei] != node:
                self.low[node] = min(self.low[node], self.dis[nei])

    def findBridge(self, graph, node):
        self.visited.add(node)
        self.dis[node], self.low[node] = self.T, self.T
        self.T += 1
        child = 0
        for nei in graph[node]:
            if nei not in self.visited:
                child += 1
                self.parents[nei] = node
                self.findAP(graph, nei)
                self.low[node] = min(self.low[node], self.low[nei])

                if node in self.parents and self.low[nei] > self.low[node]:
                    self.bridge.append((node, nei))

            elif self.parents[node] != nei:
                self.low[node] = min(self.low[node], self.dis[nei])


"----------------------------------------------------------------------------------------------------------------------"
