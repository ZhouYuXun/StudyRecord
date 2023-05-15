import sys

# 逐行讀入輸入
for line in sys.stdin:
  # 將每行字串轉成整數串列

  nums = list(line)

  # 初始化最長非遞減字串長度和目前非遞減字串長度
  longest_len = 0
  current_len = 1

  # 逐一檢查每個數字和前一個數字的大小關係
  for i in range(1, len(nums)):
    if nums[i] >= nums[i - 1]:
      current_len += 1
    else:
      # 若非遞減字串中斷，更新最長非遞減字串長度
      longest_len = max(longest_len, current_len)
      current_len = 1

  # 檢查最後一個非遞減字串的長度
  longest_len = max(longest_len, current_len)

  # 輸出最長非遞減字串長度
  print(longest_len)