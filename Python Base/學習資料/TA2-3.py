import sys

A = int(sys.stdin.read())

listonetwo = [0, 1]

while not (len(listonetwo) >= A):

  listonetwo.append(listonetwo[-1] + listonetwo[-2])

print(listonetwo[A - 1])
