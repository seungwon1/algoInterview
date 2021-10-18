"""
1. Insertion of a node in Linked List (On the basis of some constraints)
2. Delete a given node in Linked List (under given constraints)
3. Compare two strings represented as linked lists
4. Add Two Numbers Represented By Linked Lists
5. Merge A Linked List Into Another Linked List At Alternate Positions
6. Reverse A List In Groups Of Given Size
7. Union And Intersection Of 2 Linked Lists
8. Detect And Remove Loop In A Linked List
9. Merge Sort For Linked Lists
10. Select A Random Node from A Singly Linked List
"""
class listnode(object):
    def __init__(self, val):
        self.val = val
        self.next = None

# 1. Insertion of a node in Linked List (On the basis of some constraints)
def problem1(root, value):
    if not root or value <= root.val:
        node = listnode(value)
        node.next = root
        return node

    node = root
    while node.next is not None and value > node.next.val:  # 1 variable
        node = node.next

    new = Listnode(value)
    new.next, node.next = node.next, new
    return root

# 2. Delete a given node in Linked List (under given constraints)
def problem2(root, value):
    if root == value:
        if root.next is None:
            return None
        root.val = root.next.val
        root.next = root.next.next
        return root

    pointer = root
    while pointer.next is not None and pointer.next != value:
        pointer = pointer.next

    if pointer.next is None:
        return root

    if pointer.next.next is None:
        pointer.next = None
    else:
        pointer.next = pointer.next.next
    return root

# 3. Compare two strings represented as linked lists
def problem3(root1, root2):
    # it returns 0 if both strings are the same, 1 if the first linked list is lexicographically greater,
    # and -1 if the second string is lexicographically greater.
    n1, n2 = root1, root2
    while n1 and n2:
        if n1.val > n2.val:
            return 1
        elif n1.val < n2.val:
            return -1
        else:
            n1 = n1.next
            n2 = n2.next

    if not n1 and not n2:
        return 0

    return 1 if n1 else -1

# 4. Add Two Numbers Represented By Linked Lists
# allowed to modify the list
def problem4(node1, node2):
    def reverseLl(node):
        curr, cn = node, node.next
        curr.next = None
        length = 0
        while cn:
            length += 1
            tmp, cn.next = cn.next, curr
            curr, cn = cn, tmp
        return node, length

    # recursive approach
    node1, length1 = reverseLl(node1)
    node2, length2 = reverseLl(node2)

    if length1 < length2:
        node1, node2 = node2, node1

    def recursion(n1, n2, extra):
        if not n2 or not n1:
            if n1:
                v = (n1.val + extra)
                n1.val = v // 10
                n1.next = recursion(n1.next, n2, v % 10)
            else:
                return listnode(1) if extra == 1 else None

        v = (n1.val + n2.val + extra)
        n1.val = v // 10
        n1.next = recursion(n1.next, n2.next, v % 10)
        return n1

    return reverseLl(recursion(node1, node2, 0))

# w/o reversing the array
#https://leetcode.com/problems/add-two-numbers-ii/solution/

# 5. Merge A Linked List Into Another Linked List At Alternate Positions
def problem5(node1, node2):
    root = node1
    while node1 and node2:
        tmp1, node1.next = node1.next, node2
        tmp2, node2.next = node2.next, tmp1
        node1, node2 = tmp1, tmp2
    return root, node2

# 6. Reverse A List In Groups Of Given Size
# reverse every k nodes

def problem6(node1, k):
    head = node1
    if head is None:
        return head

    curr, cn = head, head.next
    curr.next = None
    length = 1
    while cn and length < k:
        tmp, cn.next = cn.next, curr
        curr, cn = cn, tmp
        length += 1

    if length < k:
        return problem6(curr, length)
    else:
        head.next = problem6(cn, k)
    return curr

# 7. Union And Intersection Of 2 Linked Lists
# approach 1: O(mn) time and O(1) space -- Brute force naive
#          2: O(mlogm + nlogn) time and O(n+m) space -- merge sort on linked list --> problem 9
# approach 3: O(m+n) time and O(m+n) space -- w/ hashing

def problem7(node1, node2):
    # approach 3
    if node1 is None:
        return None, node2

    valdict = set()
    node = node1
    union = node
    while node:
        valdict.add(node.val)
        union = node
        node = node.next

    dummy = listnode(0)
    curr = dummy
    node = node2
    while node:
        if node.val in valdict:
            curr.next = listnode(node.val)
            curr = curr.next
        else:
            union.next = listnode(node.val)
            union = union.next
        node = node.next
    return dummy.next, node1 # intersection, union

# 8. Detect And Remove Loop In A Linked List
# ------------------
#          x
#       |--|--------|
#         b     a
#       | <-- the second point of the cycle

# starting from origin and x w/ same pace
# (x+a) = (a + x) --> both points meet at the end
def problem8(head):
    p1, p2 = head, head
    while p1 and p2:
        p1 = p1.next
        p2 = p2.next
        if p2 is None:
            return False
        p2 = p2.next
        if p1 == p2:
            break

    if p2 is None:
        return False

    p1 = head
    while p1 != p2:
        p1 = p1.next
        p2 = p2.next

    p1.next = None
    return head

# 9. Merge Sort For Linked Lists
def problem9(head):
    if not head:
        return head

    def findMiddle(head):
        p1, p2 = head, head
        while p2.next:
            prev = p1
            p1 = p1.next
            p2 = p2.next.next
            if not p2:
                break
        prev.next = None
        return p1

    def mergesort(head):
        if head.next is None:
            return head

        mid = findMiddle(head)

        l, r = mergesort(head), mergesort(mid)
        return merge(l, r)

    def merge(head1, head2):
        dummy = listnode(0)
        curr = dummy
        while head1 and head2:
            if head1.val <= head2.val:
                curr.next = head1
                curr = curr.next
                head1 = head1.next
            else:
                curr.next = head2
                curr = curr.next
                head2 = head2.next
        if head2:
            curr.next = head2
        if head1:
            curr.next = head1
        return dummy.next
    return mergesort(head)

# 10. Select A Random Node from A Singly Linked List
def problem10():


    return












