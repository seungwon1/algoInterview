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
            leftSum = max(maxSum(root.left), 0)
            rightSum = max(maxSum(root.right), 0)
            self.maxval = max(self.maxval, leftSum + rightSum + root.val)
            return max(leftSum + root.val, rightSum + root.val)

        maxSum(root)
        return self.maxval


# 3. Check if a given array can represent Preorder Traversal of Binary Search Tree
class canRepresentBST(object):
    # two-pass recursive solution: O(n) time and O(n) space
    def nextGreater(self, arr):
        ans = []
        stack = []
        n = len(arr)
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] < arr[i]:
                stack.pop()
            if not stack:
                ans.append(-1)
            else:
                ans.append(stack[-1])
            stack.append(i)
        return ans[::-1]

    def canRepresentBST(self, arr):
        nextGreater = self.nextGreater(arr)
        def checkPreorder(arr, front, back, lo, hi):
            if (front > back):
                return True
            rs = nextGreater[front]
            return checkPreorder(arr, front + 1, rs - 1, lo, min(hi, arr[front])) and \
                   checkPreorder(arr, rs, back, max(lo, arr[front]), hi) and lo < arr[front] < hi
        return checkPreorder(arr, 0, len(arr) - 1, -float("inf"), float("inf"))

    # one-pass soluton w/ monotonic stack
    def canRepresentBSTonePass(self, arr):
        stack = []
        root = -float("inf")
        for n in arr:
            if n < root:
                return False
            while stack and stack[-1] < n:
                root = stack.pop()
            stack.append(n)
        return True


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
    for x in range(left, right + 1):  # range(left, right + 1):
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
    for x in range(left, right + 1):  # range(left, right + 1):
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

    def removeNodes2(root, level):  # a bit concise implementation
        if not root:
            return None
        root.left = removeNodes2(root.left, level + 1)
        root.right = removeNodes2(root.right, level + 1)
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
# recursive
def LCAbinSearchTree(root, n1, n2):
    if root.data > n1 and root.data > n2:
        return LCAbinSearchTree(root.left, n1, n2)
    if root.data < n1 and root.data < n2:
        return LCAbinSearchTree(root.right, n1, n2)
    return root

# iterative: O(H) time and O(1) space
def LCAbinSearchTreeIt(root, n1, n2):
    while root:
        if root.data < n1 and root.data < n2:
            root = root.right
        elif root.data > n1 and root.data > n2:
            root = root.left
        else:
            break
    return root

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
    return False기

# O(N) solution *** --> code failed to pass 추후 다시 시도해보
# (pre-requisite 1) to finalize the tree from the serialized string, at least two are available
# (pre-requisite 2) rolling hash or KMP can be done in linear time
def checkSubtreeInLinearTime(root1, root2):
    inorder1, preorder1 = [], []
    inorder2, preorder2 = [], []

    def dfs(root, ino, preo):
        if not root:
            return
        preo.append(str(root.val))
        preo.append(",")
        dfs(root.left)
        ino.append(str(root.val))
        ino.append(",")
        dfs(root.right)

    dfs(root1, inorder1, preorder1)
    dfs(root2, inorder2, preorder2)
    inorder1.pop()
    inorder2.pop()
    preorder1.pop()
    preorder2.pop()
    inorder1, inorder2 = "".join(inorder1), "".join(inorder2)
    preorder1, preorder2 = "".join(preorder1), "".join(preorder2)

    def isSubtree(string1, string2):
        base = 11
        mod = 2**64
        st1 = 0
        n = len(string1)
        for c in string1:
            st1 = (st1*base + ord(c)) % mod

        st2 = 0
        for i,v in enumerate(string2):
            if i >= n:
                st2 = (st2 - (string2[i-n]*(base**(n-1))) % mod) % mod
            st2 = (st1*base + ord(v)) % mod
            if st2 == st1:
                return True
        return False

    return isSubtree(inorder1, inorder2) and isSubtree(preorder1, preorder2)





# 10. Reverse alternate levels of a perfect binary tree
def reverseAlternate(root):
    def swapEven(h, dq):
        length = len(dq)
        idx = length - 1
        front, back = 0, idx
        while front < back:
            n1, n2 = dq[front][0], dq[back][0]
            n1.left, n1.right, n2.left, n2.right = n2.left, n2.right, n1.left, n1.right
            front += 1
            back -= 1

    def swapOdd(prev, h, dq):
        length = len(dq)
        idx = length - 1
        for node in prev:
            node.left = dq[idx][0]
            idx -= 1
            node.right = dq[idx][0]
            idx -= 1

    # standard BFS: O(N) time and O(N) space
    dq = deque([(root, 1)])
    prev = []
    h = 0
    while dq:
        n = len(dq)
        if h % 2 == 1 and dq:  # swap even level nodes
            swapEven(h, dq)

        for _ in range(n):
            node, h = dq.popleft()
            if h % 2 == 1:
                prev.append(node)
            if node.left:
                dq.append((node.left, h + 1))
            if node.right:
                dq.append((node.right, h + 1))

        if h % 2 == 1 and dq:  # swap odd level nodes
            swapOdd(prev, h, dq)
        prev = []
    return root