A = range(2, int(input()) + 1)

x = 0

while x < len(A):

  A = [i for i in A if i == A[x] or i % A[x] > 0]

  x += 1

print(A[-1])
