A = input()

B = int(A.split()[0])

C = A.split()[1:len(A)][B - 1]

print(C)
