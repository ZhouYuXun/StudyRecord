import sys
import collections

A = sys.stdin.read()

A = list(''.join(A))

A = collections.Counter(A)

if A['U'] - A['D'] == 0 and A['L'] - A['R'] == 0:

  print('Y')

else:

  print('N')
