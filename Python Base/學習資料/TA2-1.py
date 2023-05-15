import sys

in_txt = sys.stdin.read()

in1, in2 = in_txt.split()
in1 = int(in1)
in2 = int(in2)

i = 0
while i < in1:
  j = 0
  while j < in2:
    print(f'{i+1}x{j+1}={(i+1)*(j+1)}')
    j += 1
  i += 1
