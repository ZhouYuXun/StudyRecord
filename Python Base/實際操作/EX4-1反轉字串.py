import sys

A = sys.stdin.readline()

A = A.split()

x = 0

while x < len(A):

  A[x] = ''.join(list(reversed(A[x])))

  x += 1

A = ' '.join(A)

print(A)
