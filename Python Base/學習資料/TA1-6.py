import sys

in_txt = sys.stdin.read()

in_year = int(in_txt)

if in_year % 4 != 0:
  print(False)
elif in_year % 4 == 0 and in_year % 100 != 0:
  print(True)
elif in_year % 100 == 0 and in_year % 400 != 0:
  print(False)
elif in_year % 400 == 0:
  print(True)
else:
  print(False)
