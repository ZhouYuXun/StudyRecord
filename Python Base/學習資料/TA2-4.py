import sys

A, y = sys.stdin.read(), 0

for x in A.splitlines():

  print(f'{x}{y+1}')

  y += 1
