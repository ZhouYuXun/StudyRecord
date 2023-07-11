import sys

A = sys.stdin.read()

A, B = reversed(list(A)), []

[B.append(x) for x in A if x not in B]

B = ''.join(list(reversed(B)))

print(B)
