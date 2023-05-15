import sys

in_txt = sys.stdin.read()

if '+' in in_txt:
  in_list = in_txt.split('+')
  print(int(in_list[0]) + int(in_list[1]))
elif '-' in in_txt:
  in_list = in_txt.split('-')
  print(int(in_list[0]) - int(in_list[1]))
elif '*' in in_txt:
  in_list = in_txt.split('*')
  print(int(in_list[0]) * int(in_list[1]))
else:
  in_list = in_txt.split('/')
  print(int(int(in_list[0]) / int(in_list[1])))