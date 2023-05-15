import requests, os, zipfile
from bs4 import BeautifulSoup

from urllib.parse import urlparse

potted_flowers = ['黃金葛', '長春藤', '阿波羅千年木', '旺旺樹', '蔓綠絨', '金錢樹', '尤加利', '發財樹', '黃金葛', '長春藤', '虎尾蘭', '琴葉榕', '龜背芋', '仙人掌', '蘭花', '龍舌蘭']

for potted in potted_flowers:
  for num in range(1, 200):

    print('{}下載第{}頁中'.format(potted, num))

    # 獲取HTML資料並使用BeautifulSoup

    url = 'https://www.bing.com/images/search?q={}&form=HDRSC1&first={}'.format(potted, num)

    folder_name = '盆栽/' + potted
    response = requests.get(url)
    html = BeautifulSoup(response.text)

    # 確認資料夾是否存，如不存在則創建

    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    # 擷取包含圖片網址的內容

    for img in html.find_all('img', class_="mimg"):

      print(img)

      # 將圖片網址取出，並檢查是否為空值
      if img.get('src') != None:
        print(img.get('src'))
        # 從圖片網址取出檔案名稱，並去除無意義字串
        parsed_url = urlparse(img.get('src'))
        print(parsed_url)
        file_name = os.path.basename(parsed_url.path)
        print(file_name)
        # 檢查檔案名稱是否為空值
        if file_name != b'':
          file_name = file_name.replace("OIP.", "")+".png"
          # 用圖片網址下載圖片
          image_url = img.get('src')
          response = requests.get(image_url)
          # 結合資料夾名稱與檔案名稱
          image_path = os.path.join(folder_name, file_name)
          # 檢查檔案名稱是否存在
          if os.path.exists(image_path) != True:
            with open(image_path, "wb") as f:
              f.write(response.content)
          else:
            pass
        else:
          pass

      elif img.get('data-src') != None:
        print(img.get('data-src'))
        # 從圖片網址取出檔案名稱，並去除無意義字串
        parsed_url = urlparse(img.get('data-src'))
        print(parsed_url)
        file_name = os.path.basename(parsed_url.path)
        print(file_name)
        # 檢查檔案名稱是否為空值
        if file_name != b'':
          file_name = file_name.replace("OIP.", "")+".png"
          # 用圖片網址下載圖片
          image_url = img.get('data-src')
          response = requests.get(image_url)
          # 結合資料夾名稱與檔案名稱
          image_path = os.path.join(folder_name, file_name)
          # 檢查檔案名稱是否存在
          if os.path.exists(image_path) != True:
            with open(image_path, "wb") as f:            
              f.write(response.content)
          else:
            pass
        else:
          pass          
      else:
        pass
    