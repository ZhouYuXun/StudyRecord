# A=10 B=11 C=12 D=13 E=14 F=15 G=16 H=17 I=34 J=18 K=19 L=20
# M=21 N=22 O=35 P=23 Q=24 R=25 S=26 T=27 U=28 V=29 W=32 X=30
# Y=31 Z=33

import sys

in_txt = sys.stdin.read()

check_num = 0

if in_txt[0] == 'A':
  check_num += 1 + 0 * 9
elif in_txt[0] == 'B':
  check_num += 1 + 1 * 9
elif in_txt[0] == 'C':
  check_num += 1 + 2 * 9
elif in_txt[0] == 'D':
  check_num += 1 + 3 * 9
elif in_txt[0] == 'E':
  check_num += 1 + 4 * 9
elif in_txt[0] == 'F':
  check_num += 1 + 5 * 9
elif in_txt[0] == 'G':
  check_num += 1 + 6 * 9
elif in_txt[0] == 'H':
  check_num += 1 + 7 * 9
elif in_txt[0] == 'I':
  check_num += 3 + 4 * 9
elif in_txt[0] == 'J':
  check_num += 1 + 8 * 9
elif in_txt[0] == 'K':
  check_num += 1 + 9 * 9
elif in_txt[0] == 'L':
  check_num += 2 + 0 * 9
elif in_txt[0] == 'M':
  check_num += 2 + 1 * 9
elif in_txt[0] == 'N':
  check_num += 2 + 2 * 9
elif in_txt[0] == 'O':
  check_num += 3 + 5 * 9
elif in_txt[0] == 'P':
  check_num += 2 + 3 * 9
elif in_txt[0] == 'Q':
  check_num += 2 + 4 * 9
elif in_txt[0] == 'R':
  check_num += 2 + 5 * 9
elif in_txt[0] == 'S':
  check_num += 2 + 6 * 9
elif in_txt[0] == 'T':
  check_num += 2 + 7 * 9
elif in_txt[0] == 'U':
  check_num += 2 + 8 * 9
elif in_txt[0] == 'V':
  check_num += 2 + 9 * 9
elif in_txt[0] == 'W':
  check_num += 3 + 2 * 9
elif in_txt[0] == 'X':
  check_num += 3 + 0 * 9
elif in_txt[0] == 'Y':
  check_num += 3 + 1 * 9
else:
  check_num += 3 + 3 * 9

check_num += int(in_txt[1]) * 8
check_num += int(in_txt[2]) * 7
check_num += int(in_txt[3]) * 6
check_num += int(in_txt[4]) * 5
check_num += int(in_txt[5]) * 4
check_num += int(in_txt[6]) * 3
check_num += int(in_txt[7]) * 2
check_num += int(in_txt[8])
check_num += int(in_txt[9])

if check_num % 10 == 0:
  print('合法')
else:
  print('不合法')
