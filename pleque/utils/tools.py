def arglis(seq):
    """Returns arguments of the Longest Increasing Subsequence in the Given List/Array"""
    n = len(seq)
    p = [0] * n
    m = [0] * (n + 1)
    l = 0
    for i in range(n):
        lo = 1
        hi = l
        while lo <= hi:
            mid = (lo + hi) // 2
            if seq[m[mid]] < seq[i]:
                lo = mid + 1
            else:
                hi = mid - 1

        new_l = lo
        p[i] = m[new_l - 1]
        m[new_l] = i

        if new_l > l:
            l = new_l

    s = []
    k = m[l]

    for i in range(l - 1, -1, -1):
        s.append(k)
        k = p[k]
    return s[::-1]


def lis(seq):
    """Returns the Longest Increasing Subsequence in the Given List/Array"""
    return [seq[i] for i in arglis(seq)]


