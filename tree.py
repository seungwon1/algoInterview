"""
1. Find Minimum Depth of a Binary Tree
2. Maximum Path Sum in a Binary Tree ***
3. Check if a given array can represent Preorder Traversal of Binary Search Tree ***
4. Check whether a binary tree is a full binary tree or not
5. Bottom View Binary Tree
6. Print Nodes in Top View of Binary Tree
7. Remove nodes on root to leaf paths of length < K
8. Lowest Common Ancestor in a Binary Search Tree
9. Check if a binary tree is subtree of another binary tree ***
10. Reverse alternate levels of a perfect binary tree
"""
import collections
from collections import deque
# 1. Find Minimum Depth of a Binary Tree
# DFS: O(n) time and O(H) space for recursion stack
# We can also implement BFS for this task to improve for better time/space (the complexity remains the same).
def minDepth(root, depth):
    if not root:
        return 0

    def dfs(root, depth):
        if not root:
            return float("inf")
        if not root.left and not root.right:
            return depth
        return min(dfs(root.left, depth + 1), dfs(root.right, depth + 1))

    return dfs(root, 1)

# 2. Maximum Path Sum in a Binary Tree
# O(N) time and O(N) space
class findMaxSum(object):
    def findMS(self, root):
        self.maxval = -float("inf")
        def maxSum(root):
            if not root:
                return -float("inf")
            leftSum = maxSum(root.left)
            rightSum = maxSum(root.right)
            self.maxval = max(self.maxval, root.val, leftSum+root.val, rightSum+root.val, leftSum+rightSum+root.val)
            return max(leftSum+root.val, rightSum+root.val, root.val, 0)        
        maxSum(root)
        return self.maxval

# 3. Check if a given array can represent Preorder Traversal of Binary Search Tree
def canRepresentBST(pre):
    


    return


# 4. Check whether a binary tree is a full binary tree or not
# O(N) time and O(N) space
# optimized implementation
def isFullTree(root):
    def checkFullTree(root):
        if not root:
            return 0
        a = checkFullTree(root.left)
        if a is False:
            return False
        b = checkFullTree(root.right)
        if b is False:
            return False
        return False if a + b == 1 else 1
    return checkFullTree(root) == True

# concise implementation
def isFullTreeConcise(root):
    if not root:
        return True
    if (not root.left and root.right) or (root.left and not root.right):
        return False
    if not root.left and not root.right:
        return True
    return isFullTreeConcise(root.left) and isFullTreeConcise(root.right)

# 5. Bottom View Binary Tree
# Using external memory (e.g., dictionary) is easy approach
# W/O using external memory,
# O(N) time and O(N) space
def bottomView(root):
    hashtable = collections.defaultdict(dict)
    def dfs(root, x, y):
        if not root:
            return 0, 0, 0
        if y not in hashtable[x]:
            hashtable[x][y] = [root.data]
        else:
            hashtable[x][y].append(root.data)
        l1, r1, y1 = dfs(root.left, x - 1, y - 1)
        l2, r2, y2 = dfs(root.right, x + 1, y - 1)
        return min(l1, l2, x), max(r1, r2, x), min(y1, y2, y)

    left, right, height = dfs(root, 0, 0)
    res = []
    for x in range(left, right+1):  # range(left, right + 1):
        if len(hashtable[x]) == 0:
            continue
        y = min(hashtable[x].keys())
        res.append(hashtable[x][y][-1])
    return res

# 6. Print Nodes in Top View of Binary Tree
# O(N) time and O(N) space
def topview(root):
    hashtable = collections.defaultdict(dict)

    def dfs(root, x, y):
        if not root:
            return 0, 0, 0
        if y not in hashtable[x]:
            hashtable[x][y] = [root.data]
        else:
            hashtable[x][y].append(root.data)
        l1, r1, y1 = dfs(root.left, x - 1, y - 1)
        l2, r2, y2 = dfs(root.right, x + 1, y - 1)
        return min(l1, l2, x), max(r1, r2, x), min(y1, y2, y)

    left, right, height = dfs(root, 0, 0)
    res = []
    for x in range(left, right+1):  # range(left, right + 1):
        if len(hashtable[x]) == 0:
            continue
        y = max(hashtable[x].keys())
        res.append(hashtable[x][y][0])
    return res

# 7. Remove nodes on root to leaf paths of length < K
# O(N) time and O(N) space
def removeShortPathNodes(root, k):
    def removeNodes(root, level):
        if not root:
            return None, 0
        root.left, maxHeight1 = removeNodes(root.left, level + 1)
        root.right, maxHeight2 = removeNodes(root.right, level + 1)
        if max(maxHeight1, maxHeight2, level) < k:
            return None, max(maxHeight1, maxHeight2, level)
        return root, max(maxHeight1, maxHeight2, level)

    def removeNodes2(root, level): # a bit concise implementation
        if not root:
            return None
        root.left = removeNodes2(root.left, level+1)
        root.right = removeNodes2(root.right, level+1)
        if root.left is None and root.right is None and level < k:
            return None
        return root

    return removeNodes(root, 1)[0]  # removeNodes2(root, 1)

# 8. Lowest Common Ancestor in a Binary Search Tree
# assuming that the values of node are distinct, let n1, n2 be the given values of the two nodes.
# O(N) time and O(N) space
def lca(root, n1, n2):
    if not root:
        return False
    l = lca(root.left, n1, n2)
    if type(l) != bool:
        return l
    r = lca(root.right, n1, n2)
    if type(r) != bool:
        return r
    curr = (root.val == n1 or root.val == n2)
    if (l and curr) or (r and curr) or (l and r):
        return root
    return l or r or curr

# O(H) time and O(H) space
def LCAbinSearchTree(root, n1, n2):
    return


# 9. Check if a binary tree is subtree of another binary tree ***
# check if the first tree is subtree of the second one:
# O(N**2) time and O(N) space
def checkSubtree(root1, root2):
    if not root1:
        return True

    candidates = []
    def dfs(root):
        if not root:
            return
        if root.val == root1.val:
            candidates.append(root)
        dfs(root.left)
        dfs(root.right)

    dfs(root2)
    if len(candidates) == 0:
        return False

    def compare(node1, node2):
        if not ((node1 and node2) or (not node1 and not node2)):
            return False
        if not node1 and not node2:
            return True
        return node1.val == node2.val and compare(node1.left, node2.left) and compare(node1.right, node2.right)

    for candidate in candidates:
        if compare(root1, candidate):
            return True
    return False

# O(N) solution ***


# 10. Reverse alternate levels of a perfect binary tree
def reverseAlternate(root):
    # standard BFS: O(N) time and O(N) space
    dq = deque([(root, 1)])
    prev = deque()
    while dq:
        n = len(dq)
        for _ in range(n):
            node, h = dq.popleft()
            if h % 2 == 1:
                prev.append(node)
            if node.left:
                dq.append((node.left, h+1))
            if node.right:
                dq.append((node.right, h+1))

        if h % 2 == 1 and dq:
            length = len(dq)
            idx = length-1
            front, back = 0, idx
            while front < back:
                n1, n2 = dq[front][0], dq[back][0]
                n1.left, n1.right, n2.left, n2.right = n2.left, n2.right, n1.left, n1.right
                front += 1
                back -= 1

            for node in prev:
                node.left = dq[idx][0]
                idx -= 1
                node.right = dq[idx][0]
                idx -= 1
        prev = []
    return root

