graph = [[], [2, 5], [1, 3, 4], [2], [2], [1]]

# ver 1: Euler Tour Tree by definition
b = []
def dfs(node, p):
    for nei in graph[node]:
        if nei != p:
            b.append([node, nei])
            dfs(nei, node)
    if p:
        b.append([node, p])

print('\nrecursive ver 1')
dfs(1, 0)
print('arr', b)

st, b = [[1, 0, 0]], []
while st:
    node, par, exp = st[-1]
    if exp:
        st.pop()
        if par:
            b.append([node, par])
        continue
    if par:
        b.append([par, node])
    st[-1][-1] = 1
    for nei in graph[node][::-1]:
        if nei != par:
            st.append([nei, node, 0])

print('iterative ver 1')
print('arr', b)



# ver 2: lca app
b = []
def dfs(node, p):
    b.append(node)
    for nei in graph[node]:
        if nei != p:
            dfs(nei, node)
    if node != 1:
        b.append(p)


print('\nrecursive ver 2')
dfs(1, 0)
print('arr', b)


st, b = [[1, 0, 0]], []
while st:
    node, par, exp = st[-1]
    if exp:
        st.pop()
        if node > 1:
            b.append(par)
        continue
    b.append(node)
    st[-1][-1] = 1
    for nei in graph[node][::-1]:
        if nei != par:
            st.append([nei, node, 0])

print('iterative ver 2')
print('arr', b)



# ver 3: range sum query app
# recursive implementations

st, et = [0] * 6, [0] * 6
counter = 0
b = []

def dfs(node, p):
    global counter
    counter += 1
    st[node] = counter
    b.append(node)
    for nei in graph[node]:
        if nei != p:
            dfs(nei, node)
    et[node] = counter

print('\nrecursive ver 3')
dfs(1, 0)
print('arr', b)
print('st', st)
print('et', et)

# iterative implementations
st, et = [0] * 6, [0] * 6
counter = 0

a = [(1, 0, 0)]
b = []
while a:
    node, par, exp = a.pop()
    if exp:
        et[node] = counter
        continue
    counter += 1
    st[node] = counter
    b.append(node)
    a.append((node, par, 1))
    for nei in graph[node][::-1]:
        if nei != par:
            a.append((nei, node, 0))

print('iterative ver 3')
print('arr', b)
print('st', st)
print('et', et)
