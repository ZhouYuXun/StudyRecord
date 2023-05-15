import sys

A = list(map(int, sys.stdin.read().split()))

A = map(str, [x + 1 for x in A])

A = ' '.join(A)

print(A)
