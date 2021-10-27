"""
1. Find Minimum Depth of a Binary Tree
2. Maximum Path Sum in a Binary Tree
3. Check if a given array can represent Preorder Traversal of Binary Search Tree
4. Check whether a binary tree is a full binary tree or not
5. Bottom View Binary Tree
6. Print Nodes in Top View of Binary Tree
7. Remove nodes on root to leaf paths of length < K
8. Lowest Common Ancestor in a Binary Search Tree
9. Check if a binary tree is subtree of another binary tree
10. Reverse alternate levels of a perfect binary tree
"""

# 1. Find Minimum Depth of a Binary Tree
def minDepth(root, depth):
  if not root:
    return 0
  def dfs(root, depth):
    if not root:
      return float("inf")
    if not root.left and not root.right:
      return depth
    return min(dfs(root.left, depth+1), dfs(root.right, depth+1))
  return dfs(root, 1)

# DFS: O(n) time and O(H) space for recursion stack
# We can also implement BFS for this task to improve for better time/space (the complexity remains the same).

# 2. Maximum Path Sum in a Binary Tree
def findMaxSum(root):
   

  return

# 3. Check if a given array can represent Preorder Traversal of Binary Search Tree
def canRepresentBST(pre):

  return

# 4. Check whether a binary tree is a full binary tree or not
def isFullTree(root):
  def checkFullTree(root):
    if not root:
      return 0
    a = checkFullTree(root.left)
    b = checkFullTree(root.right)
    if a == False or b == False:
      return False
    else:
      if a + b == 1:
        return False
      else:
        return 1
  return checkFullTree(root) == True
  
# 5. Bottom View Binary Tree
# 6. Print Nodes in Top View of Binary Tree
# 7. Remove nodes on root to leaf paths of length < K
# 8. Lowest Common Ancestor in a Binary Search Tree
# 9. Check if a binary tree is subtree of another binary tree
# 10. Reverse alternate levels of a perfect binary tree












