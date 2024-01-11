class LCA:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        self.it = [0] * self.n
        self.ot = [0] * self.n
        self.d = 0
        self.ett()
        self.v = self.d.bit_length()
        self.par = [[0] * self.v for _ in range(self.n)]
        self.computePar()

    def ett(self):
        root, clk = 1, 0
        st = [[root, 0, 0, 0]]
        while st:
            node, par, d, exp = st[-1]
            if exp:
                self.ot[node] = clk
                self.d = max(self.d, d)
                st.pop()
            else:
                st[-1][-1] = 1
                clk += 1
                self.it[node] = clk
                for nei in self.graph[node]:
                    if nei != par:
                        st.append([nei, node, d+1, 0])
                        self.par[nei][0] = node

    def isAncestor(self, i, j):
        if self.it[i] <= self.it[j] <= self.ot[i]:
            return 1, i
        elif self.it[j] <= self.it[i] <= self.ot[j]:
            return 1, j
        else:
            return -1

    def computePar(self):
        for i in range(self.n):
            for j in range(self.v - 1):
                self.par[i][j + 1] = self.par[self.par[i][j]][j]

    def findLCA(self, i, j):
        # compute LCA in O(logN) time using binary lifting
        lca = self.isAncestor(i, j)
        if lca[0]:
            return lca[1]
        for d in range(self.v-1, -1, -1):
            if not self.isAncestor(self.par[i][d], j)[0]:
                i = self.par[i][d]
        return i
    
