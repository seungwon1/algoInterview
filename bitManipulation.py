import collections

"""
1. Maximum Subarray XOR
2. Magic Number
3. Sum of bit differences among all pairs
4. Swap All Odds And Even Bits
5. Find the element that appears once
6. Binary representation of a given number
7. Count total set bits in all numbers from 1 to n
8. Rotate bits of a number
9. Count number of bits to be flipped to convert A to B
10. Find Next Sparse Number
"""


class Trienode():
    def __init__(self, val=0):
        self.val = val
        self.children = {}


# 1. Maximum Subarray XOR
# naive approach: iterate over all pairs of (i,j) s.t. 0<= i <= j <= n
# efficient solution: O(N) time
# for finding max xor pair, insert each element in Trie and query the element for each idx
# for finding max xor sub-array, insert prefix xor in Trie and query the prefix xor for each idx
def maxSubarrayXOR(arr, n):
    root = Trienode()
    prefix = [0]
    curr = 0
    ans = arr[0]
    for num in arr:
        curr ^= num
        prefix.append(curr)

    # insert prefix into Trie as binary representation
    prefixBit = [[(num >> i) & 1 for i in range(32)][::-1] for num in prefix]
    for num in prefixBit:
        node = root
        for bit in num:
            if bit not in node.children:
                node.children[bit] = Trienode(bit)
            node = node.children[bit]

    # query each prefixSum
    curr = 0
    for num in arr:
        curr ^= num
        q = [(num >> i) & 1 for i in range(32)][::-1]
        node = root
        res = 0
        for i in range(32):
            if q[i] ^ 1 in node.children:
                node = node.children[q[i] ^ 1]
                res = (res << 1) + 1
            else:
                node = node.children[q[i]]
                res = res << 1
        ans = max(res, ans)
    return ans


# 2. Magic Number
# O(logN) time and constant space
def nthMagicNo(n):
    binN = bin(n)[2:]
    res = 0
    for c in binN:
        if c == "1":
            res = res * 5 + 5  # or 1
        else:
            res *= 5
    return res


# 3. Sum of bit differences among all pairs
# naive BF takes O(n**2) time and O(1) space
# efficient approach: O(n) time !
def sumBitDifferences(arr, n):
    # efficient approach: O(n) time
    ans = 0
    m = 10 ** 9 + 7
    for i in range(32):
        count = 0
        for j in range(n):
            if arr[j] & 1 == 0:
                count += 1
            arr[j] >>= 1
        ans += count * (n - count) * 2
    return ans % m


# 4. Swap All Odds And Even Bits: solution 참조, skip
def swapBits(x):
    even_bits = x & 0xAAAAAAAA
    odd_bits = x & 0x55555555
    even_bits >>= 1
    odd_bits <<= 1
    return even_bits | odd_bits


# 5. Find the element that appears once
# should be done in O(n) time and O(1) space, solution 참조, skip
def getSingle(arr, n):
    # 1
    mask1, mask2 = 0, 0
    for num in arr:
        mask1 = ~mask2 & (mask1 ^ num)
        mask2 = ~mask1 & (mask2 ^ num)

    # 2
    ones, twos = 0, 0
    for i in range(n):
        twos = twos ^ (ones & arr[i])
        ones = ones ^ arr[i]
        common_bit_mask = ~(ones & twos)
        ones &= common_bit_mask
        twos &= common_bit_mask
    return ones


# 6. Binary representation of a given number
# O(logN) time and O(logN) space
def binRep(n):
    ans = []
    while n:
        ans.append(str(n & 1))
        n >>= 1
    return "".join(ans[::-1])


# 7. Count total set bits in all numbers from 1 to n
# O(logN) time and O(logN) space
def countSetBits(n):
    def findSignificantBit(n):
        k = 0
        while n > 1:
            n >>= 1
            k += 1
        return k

    def countTotalBits(n):
        if n <= 1:
            return n
        k = findSignificantBit(n)
        return k * pow(2, k - 1) + n - pow(2, k) + 1 + countTotalBits(n - pow(2, k))

    return countTotalBits(n)


# 8. Rotate bits of a number: ?
# constant time and space
def rotate(n, d, left=True):
    nBits = 32
    return n << d | n >> (nBits - d) if left else n >> d | n << (nBits - d)


# 9. Count number of bits to be flipped to convert A to B
# O(logN) time and O(1) space
def flippedCount(a, b):
    res = 0
    n = a ^ b
    while n > 0:
        res += n & 1
        n >>= 1
    return res


# 10. Find Next Sparse Number
# O(logN) time and O(logN) space
def nextSparse(x):
    digits = []
    num = x
    while num > 0:
        digits.append(num & 1)
        num >>= 1
    digits.append(0)
    n = len(digits)
    lastFinal = 0
    for i in range(1, n-1):
        if digits[i-1] == digits[i] == 1 and digits[i+1] == 0:
            digits[i+1] = 1
            for j in range(i, lastFinal-1, -1):
                digits[j] = 0
            lastFinal = i + 1
    ans = 0
    for i in range(n):
        ans += digits[i] * (1 << i)
    return ans


