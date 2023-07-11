A = input()

B = list(map(int, A.split()))

if B[1] > B[0]:
  C = B[1] - B[0]
else:
  C = 200 - B[0] + B[1]

print(C)