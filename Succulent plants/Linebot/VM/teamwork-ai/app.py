from flask import Flask, request, abort  #引用Web Server套件

import json, urllib.request, os, io, sys, requests

from datetime import datetime

from linebot import (LineBotApi, WebhookHandler)

from linebot.exceptions import (InvalidSignatureError) # 引用無效簽章錯誤

from linebot.models import (
    MessageAction, URIAction, PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction, QuickReply, QuickReplyButton
)

from linebot.models import (
    TextMessage, ImageMessage, AudioMessage, VideoMessage, RichMenu
)

from linebot.models import (
    TextSendMessage, ImageSendMessage, TemplateSendMessage
)

from linebot.models.template import ButtonsTemplate

from linebot.models.events import (
    FollowEvent, MessageEvent, PostbackEvent,
)

from google.cloud import storage
from generate_signed_urls import generate_signed_url
import pymysql

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_v3 import mobilenet_v3_large

import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

# 建立line event log，用來記錄line event
client = google.cloud.logging.Client()
bot_event_handler = CloudLoggingHandler(client,name="line_bot_event")
bot_event_logger=logging.getLogger('line_bot_event')
bot_event_logger.setLevel(logging.INFO)
bot_event_logger.addHandler(bot_event_handler)

#========== 設定值 ==========
with open('env.json') as f:
    env = json.load(f)

access_token = env['YOUR_CHANNEL_ACCESS_TOKEN']
handler = WebhookHandler(env['YOUR_CHANNEL_SECRET'])
bucket_name = env['YOUR_CHANNEL_BUCKET_NAME']  #cloud storage 值區名稱
cs_sign = env['YOUR_SERVICE_ACCOUNT']

db_settings = {
    "host": env['YOUR_SQL_HOST'],
    "port": int(env['YOUR_SQL_PORT']),
    "user": env['YOUR_SQL_USER'],
    "password": env['YOUR_SQL_PASSWD'],
    "db": env['YOUR_SQL_DB'],
    "charset": "utf8"
}

g_KID = 'g_KID'
g_usertxt=" "
app = Flask(__name__)

line_bot_api = LineBotApi(access_token)

# 設定雲端資料
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

#========== 啟動server對外接口，使Line能丟消息進來 ==========
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']     #get X-Line-Signature header value
    body = request.get_data(as_text=True)  #get request body as text

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

#========== 告知handler，如果收到用戶關注FollowEvent(加好友、封鎖、解封鎖)，則做下面的方法處理 ==========
@handler.add(FollowEvent)
def reply_text_and_get_user_profile(event):
 
    # 取個資
    line_user_profile = line_bot_api.get_profile(event.source.user_id)

    # 跟line 取回用戶大頭照片，並放置在本地端
    file_name = line_user_profile.user_id + '.jpg'
    urllib.request.urlretrieve(line_user_profile.picture_url, file_name)

    global storage_client,bucket,bucket_name
    # 依照用戶id當資料夾名稱，大頭照名稱為user_pic.png
    destination_blob_name = f"{line_user_profile.user_id}/user_pic.png"
    source_file_name = file_name

    # 上傳至cloud storage
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(file_name)

    # 若用戶不存在，將用戶資料新增至mysql，table name為line_user，主鍵為user_id
    try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "SELECT count(*) FROM line_user WHERE user_id = %s"
            cursor.execute(cmd, (line_user_profile.user_id))
            result = cursor.fetchone()
            if result[0] > 0 :
              cmd = "UPDATE line_user SET picture_url = %s,display_name = %s,status_message = %s WHERE user_id = %s"
              cursor.execute(cmd, ( f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}", line_user_profile.display_name, line_user_profile.status_message, line_user_profile.user_id))
            else:
              cmd = "INSERT INTO line_user(user_id, picture_url, display_name, status_message) "
              cmd +="VALUES(%s, %s, %s, %s)"
              cursor.execute(cmd, (line_user_profile.user_id, f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}", line_user_profile.display_name, line_user_profile.status_message))

            conn.commit()
            conn.close()
            # line_bot_api.reply_message(event.reply_token, TextSendMessage(text="個資已取"))
    except Exception as ex:
        print(ex)

    # 回覆文字消息與圖片消息
    line_bot_api.reply_message(event.reply_token,
        [TextSendMessage('$點選辨識，認識一下多肉植物吧~',emojis=[{"index":0,"productId":"5ac1bfd5040ab15980c9b435","emojiId":"105"}])] )

#========== 創建QuickReplyButton ==========
# 點擊後，以用戶身份發送文字消息--MessageAction
textQuickReplyButton = QuickReplyButton(action=MessageAction(label="發送文字消息", text="text2"))

# 點擊後，彈跳出選擇時間之視窗--DatetimePickerAction
dateQuickReplyButton = QuickReplyButton(
    action=DatetimePickerAction(label="日期選擇", data="waterdate", mode="date"))

# 點擊後，開啟相機--CameraAction
cameraQuickReplyButton = QuickReplyButton(action=CameraAction(label="拍照"))

# 點擊後，切換至照片相簿選擇
cameraRollQRB = QuickReplyButton(action=CameraRollAction(label="選擇照片"))

# 點擊後，跳出地理位置
locationQRB = QuickReplyButton(action=LocationAction(label="地理位置"))

# 點擊後，以Postback事件回應Server
postbackQRB01 = QuickReplyButton(action=PostbackAction(label="創建養成植物", data="qbtn01"))
postbackQRB02 = QuickReplyButton(action=PostbackAction(label="修改資料", data="qbtn02"))

postbackQRB11 = QuickReplyButton(action=PostbackAction(label="修改植物資料", data="qbtn11"))
postbackQRB12 = QuickReplyButton(action=PostbackAction(label="查看照護紀錄", data="qbtn12"))
postbackQRB13 = QuickReplyButton(action=PostbackAction(label="查看照顧須知", data="qbtn13"))

postbackQRB21 = QuickReplyButton(action=PostbackAction(label="創建養成植物", data="qbtn21"))
postbackQRB22 = QuickReplyButton(action=PostbackAction(label="修改資料", data="qbtn22"))
postbackQRB23 = QuickReplyButton(action=PostbackAction(label="修改植物資料", data="qbtn23"))
postbackQRB24 = QuickReplyButton(action=PostbackAction(label="查看照護紀錄", data="qbtn24"))
postbackQRB25 = QuickReplyButton(action=PostbackAction(label="查看照顧須知", data="qbtn25"))

# 以QuickReply封裝該些QuickReply Button
quickReplyList0 = QuickReply(
    items = [cameraQuickReplyButton,cameraRollQRB]
)

quickReplyList1 = QuickReply(
    items = [postbackQRB01,postbackQRB02]
)

quickReplyList2 = QuickReply(
    items = [postbackQRB11,postbackQRB12,postbackQRB13]
)

quickReplyList3 = QuickReply(
    items = [postbackQRB21,postbackQRB22,postbackQRB23,postbackQRB24,postbackQRB25]
)

#========== 圖文選單設定檔：設定圖面大小、按鍵名與功能 ==========

# 設定 headers，輸入你的 Access Token，記得前方要加上「Bearer 」( 有一個空白 )
headers = {'Authorization':'Bearer '+access_token,'Content-Type':'application/json'}  

# 讀取功能說明 introduce.txt
blob = bucket.blob("introduce.txt")
with blob.open("r") as f:
    str1 = f.read()

body={
  "size": {"width": 800,"height": 270},
  "selected": True,
  "name": "richmenu1",
  "chatBarText": "〔選單顯示切換〕",
  "areas": [
    { #辨識
      "bounds":{"x":0,"y":0,"width":266,"height":270},
      "action":{"type":"postback","data":"richmenuA"}
    },
    { #養成
      "bounds":{"x":266,"y":0,"width":267,"height":270},
      "action":{"type":"postback","data":"richmenuB"} 
    },
    { #功能說明
      "bounds":{"x":533,"y":0,"width":267,"height":270},
      "action":{"type":"message","text":"%s"%str1} 
    }
  ]
}

# 向指定網址發送 request
req = requests.request('POST','https://api.line.me/v2/bot/richmenu',headers=headers,data=json.dumps(body).encode('utf-8'))
richmenu_id=req.text[15:len(req.text)-2]

blob = bucket.blob("richmenu.jpg")
with blob.open("rb") as f:
    line_bot_api.set_rich_menu_image(richmenu_id, 'image/jpeg', f.read())

headers = {'Authorization':'Bearer '+access_token}
req2 = requests.request('POST', 'https://api.line.me/v2/bot/user/all/richmenu/'+richmenu_id, headers=headers)

#========== 用戶點擊button後，觸發postback event，對其回傳做相對應處理 ==========
@handler.add(PostbackEvent)
def handle_post_message(event):
    global bucket_name,cs_sign,g_KID
    currdate = datetime.now().date()
    lineuser_ID = event.source.user_id

    #圖文選單-辨識
    if (event.postback.data.find('richmenuA') == 0):
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            "上傳照片後，會進行AI辨識，然後顯示多肉植物名稱以供確認。",quick_reply=quickReplyList0))
      
    #圖文選單-養成       
    elif (event.postback.data.find('richmenuB') == 0):        
         line_bot_api.reply_message(event.reply_token,TextSendMessage("創建養成植物：從上傳的圖片中選擇要種植的植物。\n修改資料：修改已種植的植物資料。",quick_reply=quickReplyList1))
    #按鈕--創建養成植物qbtn01--圖片輪播
    elif (event.postback.data.find('qbtn01') == 0):   
        #抓取資料檔(user_plants.csv)用戶上傳的圖片
        img_carousel = {
            "type": "template",
            "altText": "this is a image carousel template",
            "template": {
                "type": "image_carousel",
                "columns": []
            }}

        j = 0  # 輪播圖片最多10筆
        # try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "SELECT count(*) FROM user_plants WHERE UID = %s and toplant != 'Y'"
            cursor.execute(cmd, (lineuser_ID))
            result = cursor.fetchone()
            if result[0] > 0:
                cmd = "SELECT KID,name,toplant,picfilenm FROM user_plants WHERE UID = %s and toplant != 'Y' ORDER BY plantdate DESC LIMIT 10"
                cursor.execute(cmd, (lineuser_ID))
                result = cursor.fetchmany(10) #取10筆

                for i in range(0,len(result)):
                    filepath = lineuser_ID+'/pic1/'+result[i][3]
                    furl = generate_signed_url(cs_sign,bucket_name,filepath)

                    img_carousel['template']['columns'].append({
                            "imageUrl": furl,
                            "action": {
                            "type": "postback",
                            "label": f"{result[i][1]}栽種請點選",
                            "data": f"imgcar1,{result[i][0]},{result[i][1]}"
                            }
                        }
                    )
                line_bot_api.reply_message(event.reply_token, TemplateSendMessage.new_from_json_dict(img_carousel))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage("沒有尚未種植的圖檔！"))
        conn.close()
        # except Exception as ex:
        #     print(ex)

    #確認辨識結果，輪播連結imgcar0
    elif (event.postback.data.find('imgcar0') == 0):
        str1 = event.postback.data.split(',')

        # 將資料新增至user_plants.csv
        # try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "INSERT INTO user_plants(KID,PID,UID,name,identdate,picfilenm,toplant) "
            cmd += "VALUES(%s, %d, %s, %s, %s, %s, %s)"
            cursor.execute(cmd, (event.message.id,int(str1[1]),lineuser_ID,str1[2],currdate,f'{str1[3]}.jpg','N'))

            conn.commit()
            conn.close()
        # except Exception as ex:
        #     print(ex)

    #輪播連結imgcar1是否種植
    elif (event.postback.data.find('imgcar1') == 0):
        str1 = event.postback.data.split(',')
        # 種植否=Y、種植日期存檔
        # try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "UPDATE user_plants SET toplant = 'Y',plantdate = %s "
            cmd += "WHERE UID = %s AND KID = %s"
            cursor.execute(cmd, (currdate,lineuser_ID,str1[1]))

            conn.commit()
            conn.close()
        # except Exception as ex:
        #     print(ex)

        line_bot_api.reply_message(event.reply_token, TextSendMessage(f"{str1[2]}設定為已種植"))
    #按鈕--修改資料qbtn02--圖片輪播
    elif (event.postback.data.find('qbtn02') == 0):
        #抓取資料檔(user_plants.csv)用戶以養植的圖片
        img_carousel = {
            "type": "template",
            "altText": "this is a image carousel template",
            "template": {
                "type": "image_carousel",
                "columns": []
            }}

        j = 0  # 輪播圖片最多10筆
        # try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "SELECT COUNT(*) FROM user_plants WHERE UID = %s and toplant = 'Y' "
            cursor.execute(cmd, (lineuser_ID))
            result = cursor.fetchone()
            if result[0] > 0:
                cmd = "SELECT KID,nickname,plantdate,picfilenm FROM user_plants WHERE UID = %s and toplant = 'Y' ORDER BY plantdate DESC LIMIT 10"
                cursor.execute(cmd, (lineuser_ID))
                result = cursor.fetchmany(10) #取10筆

                for i in range(0,len(result)):
                    filepath = lineuser_ID+'/pic1/'+result[i][3]
                    furl = generate_signed_url(cs_sign,bucket_name,filepath)
                    labelstr = f"修改{result[i][1]}{result[i][2]}"
                    img_carousel['template']['columns'].append({
                            "imageUrl": furl,
                            "action": {
                            "type": "postback",
                            "label": labelstr[0:13],
                            "data": f"imgcar2,{result[i][0]},{result[i][1]},{result[i][2]}"
                            }
                        }
                    )
                line_bot_api.reply_message(event.reply_token, TemplateSendMessage.new_from_json_dict(img_carousel))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage("沒有已養植的多肉植物！"))
        conn.close()
        # except Exception as ex:
        #     print(ex)
    #輪播連結imgcar2 已養植多肉
    elif (event.postback.data.find('imgcar2') == 0):
        str1 = event.postback.data.split(',')
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            f"目前選擇：{str1[0]}{str1[1]}{str1[2]}",quick_reply=quickReplyList2))

        g_KID = str1[1]
        bot_event_logger.info(g_KID)
    #快速按鈕--修改植物資料
    elif (event.postback.data.find('qbtn11') == 0):
        pass
    # 快速按鈕--查看照護記錄
    elif (event.postback.data.find('qbtn12') == 0):
        pass
    #快速按鈕--查看照顧須知
    elif (event.postback.data.find('qbtn13') == 0):
        # try:
        conn = pymysql.connect(**db_settings)
        with conn.cursor() as cursor:
            cmd = "SELECT COUNT(*) FROM plants a,user_plants b WHERE a.PID=b.PID AND b.KID=%s"
            cursor.execute(cmd, (g_KID))
            bot_event_logger.info(g_KID)
            result = cursor.fetchone()
            if result[0] > 0:
                cmd = "SELECT a.name,a.water,a.sunshine,a.fertilize,a.pest,a.notices "
                cmd += "FROM plants a,user_plants b WHERE a.PID=b.PID AND b.KID=%s"
                cursor.execute(cmd, (g_KID))
                bot_event_logger.info('g_KID')
                result = cursor.fetchone()
                str1 = f"【名稱】{result[0]}/n【澆水】{result[1]}/n【光照】{result[2]}/n【施肥】{result[3]}/n【病蟲害】{result[4]}/n【注意事項】{result[5]}/n"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(str1))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage("沒有已養植的多肉植物！"))
        conn.close()
        # except Exception as ex:
        #     print(ex)
    else:
        pass

# 用戶發出文字消息時， 按條件內容, 回傳文字消息
@handler.add(MessageEvent, message=TextMessage)
def handle_messageT(event):
  global g_usertxt
  g_usertxt=event.message.text
  
  #@暱稱
  if(event.message.text.find('@') != -1):
    pass


'''
#========== 若收到圖片消息時，先回覆用戶文字消息，並從Line上將照片拿回。==============
'''
@handler.add(MessageEvent, message=ImageMessage)
def handle_messageI(event):
    
    lineuser_ID = event.source.user_id
    message_content = line_bot_api.get_message_content(event.message.id)

    #### 辨識 ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #將圖檔存到cloud storage
    filepath = f'{lineuser_ID}/pic1/{event.message.id}.jpg'
    blob = bucket.blob(filepath)
    with blob.open('wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)

            file_name = event.message.id + '.jpg'
            with open(file_name, 'wb') as fd2:
                for chunk in message_content.iter_content():
                    fd2.write(chunk)

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='圖片已上傳'+ ' ' + event.message.id))
            message_content = line_bot_api.get_message_content(event.message.id)

            # load image
            # img_path = event.message.id+'.jpg'
            # img_path = generate_signed_url(cs_sign, bucket_name, filepath)
            # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            # img = Image.open(io.BytesIO(chunk)).convert("RGB")
            img = Image.open(fd2.read(),'jpg')

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    '''
    載入辨識訓練的csv檔
    '''
    class_dict = {}
    pic_dict = {}

    # try:
    conn = pymysql.connect(**db_settings)
    with conn.cursor() as cursor:
        cmd = "SELECT * FROM plants "
        cursor.execute(cmd,)
        result = cursor.fetchall()
        if len(result) > 0:
            for data in result:
                (key, val, picfilenm) = data[0], data[1], data[8]
                class_dict[int(key)] = val
                pic_dict[int(key)] = picfilenm
            # print(class_dict)
        conn.close()
    # except Exception as ex:
    #     print(ex)

    # create model
    model = mobilenet_v3_large(num_classes=22).to(device)
    # load model weights
    model_weight_path = "MobileNetV3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # dict probability : category id
    dict_result = dict()

    for i in range(len(predict)):
        prob = predict[i].numpy()
        if prob > 0.1:
            dict_result[float(prob)] = i

    # top 3 category (probability > 10%)
    result_list = list()

    for i in sorted(dict_result.keys(), reverse=True)[:3]:
        result_list.append(dict_result[i])
    # retrun top 3 category (probability > 10%)
    # ans = result_list

    # for i, id in enumerate(ans):
    #     print(f"第{i + 1}可能:", class_dict[f"{id}"])

    #辨識結果輪播
    img_carousel = {
        "type": "template",
        "altText": "辨識結果輪播",
        "template": {
            "type": "image_carousel",
            "columns": []
        }
    }

    for i in range(0, len(result_list)):
        filepath = f'/pic/{pic_dict[result_list[i]]}'
        furl = generate_signed_url(cs_sign, bucket_name, filepath)
        img_carousel['template']['columns'].append({
            "imageUrl": furl,
            "action": {
                "type": "postback",
                "label": f"是否為{class_dict[i]}?",
                "data": f"imgcar0,{result_list[i]},{class_dict[i]},{event.message.id}.jpg"
            }
        })

        if i >= 10: #輪播最多10張圖
            break
        if i <= 0:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("無辨識結果！"))
        else:
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage.new_from_json_dict(img_carousel))

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))