import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model_v3 import mobilenet_v3_large


def predict(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         
    # path


    # load image
    # img_path = "b638bcb4d978ce25d2980b22ce1c51a6-max-w-1024.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = img.convert("RGB")

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    model = mobilenet_v3_large(num_classes=27).to(device)
    # load model weights
    model_weight_path = "MobileNetV3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()

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
    print(type(result_list), result_list)
    return result_list
    


# if __name__ == '__main__':
    
#     with open("class_indices.json", "r") as f:
#         class_indict = json.load(f)
#     ans = main()

#     for i, id in enumerate(ans):
#         print(f"第{i + 1}可能:",class_indict[f"{id}"])
