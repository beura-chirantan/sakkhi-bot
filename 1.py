def max_sculptures(N, M, C, weights):
    def can_pack(k):
        # Try to pack k sculptures (from the rightmost k)
        cnt = 1
        curr = 0
        for w in weights[N - k:]:
            if w > C:
                return False
            if curr + w > C:
                cnt += 1
                curr = 0
            curr += w
        return cnt <= M

    left, right = 0, N
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if can_pack(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans

# Example input
N, M, C = 3, 2, 6
weights = [4, 2, 5]
print(max_sculptures(N, M, C, weights))






















