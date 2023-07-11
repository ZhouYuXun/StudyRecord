import sys

A = int(sys.stdin.read())

num = 0

while A != 0:

  if A % 2 != 0:

    A -= 1

    num += 1

  else:

    A = A / 2

    num += 1

print(num)