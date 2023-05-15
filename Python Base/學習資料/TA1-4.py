A = input()

B = A.split(',')

C = B[0].split()
D = B[1].split()

count = 0

for X in range(0, 5):
  if C[X] in D:
    count += 1

if count < 3:
  print(0)
elif count == 3:
  print(100)
elif count == 4:
  print(1000)
else:
  print(10000)
