class Tree:
    def __init__(self, graph, op=None, root=1, lift=1):
        self.graph = graph
        self.n = len(graph)
        self.n0 = self.n.bit_length()
        self.op = op
        self.root = root
        self.depth = [0] * self.n
        self.par = [0] * self.n
        self.it, self.ot = [0] * self.n, [0] * self.n
        self.ot[0] = self.n
        self.ett = []
        self.walk = []
        self.st = []
        self.dfs(lift)

    def dfs(self, lift):
        st = [[self.root, 0, 0, 0]]
        clk = 0
        while st:
            node, p, d, exp = st[-1]
            if exp:
                st.pop()
                self.depth[node] = d
                self.ot[node] = clk
                if node != self.root:
                    self.ett.append(self.par[node])
                continue
            st[-1][-1] = 1
            clk += 1
            self.it[node] = clk
            self.ett.append(node)
            for nei in self.graph[node]:
                if nei != p:
                    st.append([nei, node, d+1, 0])
                    self.par[nei] = node
        if lift:
            self.computeWalk()

    def computeWalk(self):
        self.walk = [[0] * self.n0 for _ in range(self.n)]
        for j in range(self.n0):
            for i in range(1, self.n):
                self.walk[i][j] = self.walk[self.walk[i][j-1]][j-1] if j else self.par[i]

    def dist(self, i, j):
        return self.depth[i] + self.depth[j] - 2 * self.depth[self.findLCA(i, j)]

    def isAncestor(self, i, j):
        return self.it[i] <= self.it[j] <= self.ot[i]

    def findLCA(self, i, j):
        if self.isAncestor(i, j):
            return i
        elif self.isAncestor(j, i):
            return j
        else:
            for step in range(self.n0-1, -1, -1):
                if self.walk[i][step] == 0 or self.isAncestor(self.walk[i][step], j):
                    continue
                i = self.walk[i][step]
            return self.walk[i][0]

    def computeSpt(self):
        l = (len(self.ett)-1).bit_length()
        self.st = [[-1] * len(self.ett) for _ in range(l)]
        for j in range(l):
            for i in range(len(self.ett)):
                self.st[j][i] = self.ett[i] if j == 0 else self.op(self.st[j-1][i], self.st[j-1][i + 1 << (j-1)])
        