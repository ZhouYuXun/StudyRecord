A = input()

B = A.split()

C = ['星期五', '星期六', '星期日']

if B[0] in C:
  print('不開市')
else:
  print(int(B[1]) * int(B[2]))
