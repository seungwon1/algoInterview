import sys
input = sys.stdin.readline
from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

def C(x, y):
    return 0

N, inf = 35001, 3500000
a = [0] * N
dp0 = [inf] * (N+1)
dp1 = [inf] * (N+1)
@bootstrap
def comp(lo, hi, il, ir):
    if lo > hi:
        yield 1
    mid = (lo + hi) >> 1
    best_idx = -1
    val = inf
    for j in range(il, min(mid, ir)+1):
        if dp0[j-1] + C(j, mid) <= val:
            val = dp0[j-1] + C(j, mid)
            best_idx = j
    dp1[mid] = val
    yield comp(lo, mid-1, il, best_idx)
    yield comp(mid+1, hi, best_idx, ir)
