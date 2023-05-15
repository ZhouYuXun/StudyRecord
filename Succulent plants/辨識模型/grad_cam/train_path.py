import os
import sys
import json

from comet_ml import Experiment
from torch import Tensor
from torchsampler import ImbalancedDatasetSampler


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm


from model_v3 import mobilenet_v3_large
from comet_ml.integration.pytorch import log_model

# class weights
from collections import Counter 
# experiment = Experiment(
#   api_key = "oRjh6XGWKwj5us9cwCQA1arVk",
#   project_name = "classfiy-project",
#   workspace="teams-project",
#   display_summary_level=0,
# )
experiment = Experiment()


class ConfusionMatrixCallbackReuseImages():
    def __init__(self, experiment, inputs, targets, confusion_matrix):
        self.experiment = experiment
        self.inputs = inputs
        self.targets = targets
        self.confusion_matrix = confusion_matrix

    def on_epoch_end(self, epoch, logs={}):
        predicted = self.model.predict(self.inputs)
        self.confusion_matrix.compute_matrix(self.targets, predicted, images=self.inputs)
        self.experiment.log_confusion_matrix(
            matrix=self.confusion_matrix,
            title="Confusion Matrix, Epoch #%d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )


# def _is_tensor_a_torch_image(x: Tensor) -> bool:
#     return x.ndim >= 2

# def _assert_image_tensor(img: Tensor) -> None:
#     if not _is_tensor_a_torch_image(img):
#         raise TypeError("Tensor is not a torch image.")

# def unnormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
#     _assert_image_tensor(tensor)

#     if not tensor.is_floating_point():
#         raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

#     if tensor.ndim < 3:
#         raise ValueError(
#             f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
#         )

#     if not inplace:
#         tensor = tensor.clone()

#     dtype = tensor.dtype
#     mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
#     std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
#     if (std == 0).any():
#         raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
#     if mean.ndim == 1:
#         mean = mean.view(-1, 1, 1)
#     if std.ndim == 1:
#         std = std.view(-1, 1, 1)
#     return tensor.mul_(mean).add_(std)
        


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    

    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 16,
    "num_classes": 27,
    "alpha": 1,
    "input_size": 224,
    "mobilenet_name": mobilenet_v3_large,
    "loss_function_name": nn.CrossEntropyLoss,
    "optimizer_name": optim.Adam,
    "ImbalancedDatasetSampler": False, # ImbalancedDatasetSampler or class_weights
    "class_weights": False
    }
    assert not (hyper_params["ImbalancedDatasetSampler"] and hyper_params["class_weights"]),"both_True"
    model_name_set="mobilenet_v3_large"
    experiment.log_parameters(hyper_params)
    experiment.set_name(model_name_set + "b" + str(hyper_params["batch_size"]) + "e" + str(hyper_params["epochs"]) )
    
    experiment.add_tags([model_name_set, "CrossEntropyLoss", "optim.Adam", "class wight"])

    # model
    model_name_set="mobilenet_v3_large"
    save_name = './{}_{}_{}_{}.pth'.format(hyper_params["batch_size"], hyper_params["epochs"], 
                            hyper_params["alpha"], model_name_set)

    # path_load
    load_path = "/content/drive/MyDrive/model/deep-learning-for-image-processing-master/pytorch_classification/Test6_mobilenet/path.json"
    with open(load_path, "r", encoding="utf-8") as f:
        path_load = json.load(f)


    classjson_path = path_load["classjson_path"]
    root_path = path_load["root_path"]
    data_set_dir = path_load["data_set_dir"]
    flower_dir = path_load["flower_dir"]
    weight_path = path_load["weight_path"]
    train_save_path = path_load["train_save_path"] + save_name
    
    


    # ToTensor H，W，C ——> C，H，W C/255 [0~1]
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(hyper_params["input_size"]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(hyper_params["input_size"]+32),
                                   transforms.CenterCrop(hyper_params["input_size"]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(hyper_params["input_size"]+32),
                                   transforms.CenterCrop(hyper_params["input_size"]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), root_path))  # get data root path
    image_path = os.path.join(data_root, data_set_dir, flower_dir)  # flower data set path
    
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    

    # class weights
    if hyper_params["class_weights"]:
        label_nums_dic = Counter([j for i, j in train_dataset.imgs])
        sample_num_list = label_nums_dic.values()
        max_sample_num = max(label_nums_dic.values())
        weights = [max_sample_num/label_nums for label_nums in sample_num_list]
        class_weights = torch.FloatTensor(weights).to(device)




    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())


    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(classjson_path, 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), hyper_params["batch_size"] if hyper_params["batch_size"] > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    if hyper_params["ImbalancedDatasetSampler"]:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=ImbalancedDatasetSampler(train_dataset),
                                                batch_size=hyper_params["batch_size"],
                                                num_workers=nw)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=hyper_params["batch_size"],
                                                shuffle=True,
                                                num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=hyper_params["batch_size"], shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = hyper_params["mobilenet_name"](num_classes=hyper_params["num_classes"], alpha=hyper_params["alpha"])
    # net = mobilenet_v3_large(num_classes=hyper_params["num_classes"], alpha=hyper_params["alpha"])
    # Initialize and train your model
    # model = TheModelClass()
    # train(model)

    # Seamlessly log your Pytorch model
    
    log_model(experiment, net, model_name=model_name_set)
    if weight_path != "":
        # load pretrain weights
        # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        model_weight_path = weight_path
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        
        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

        # freeze features weights
        # for param in net.features.parameters():
        #     param.requires_grad = False

    net.to(device)
    
    # define loss function
    # class weight
    if hyper_params["class_weights"]:
        loss_function = hyper_params["loss_function_name"](weight=class_weights)
    else:
        loss_function = hyper_params["loss_function_name"]()
    
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = hyper_params["optimizer_name"](params, lr=hyper_params["learning_rate"])
    best_net = hyper_params["mobilenet_name"](num_classes=hyper_params["num_classes"], alpha=hyper_params["alpha"])
    best_acc = 0.0
    save_path = train_save_path
    train_steps = len(train_loader)

    # Print model's state_dict
    modelstr=""
    for param_tensor in net.state_dict():
        modelstr += str(param_tensor) + ", \t" + str(net.state_dict()[param_tensor].size()) + "\n"
    experiment.set_model_graph(modelstr,overwrite=False)
    
        

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])



    for epoch in range(hyper_params["epochs"]):
        # train
        net.train()
        running_loss = 0.0
        correct = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            train_r = torch.max(logits, dim=1)[1]
            batch_correct = torch.eq(train_r,labels.to(device)).sum().item()
            batch_total = labels.size(0)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            

            # log traing_batch_loss
            experiment.log_metric("train_batch_loss", loss.item())
            
            # log train_batch_accuracy
            experiment.log_metric("train_batch_accuracy", batch_correct / batch_total)

            # print statistics
            running_loss += loss.item()
            correct += batch_correct

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     hyper_params["epochs"],
                                                                     loss)
            
        experiment.log_metric("train_accuracy", correct / train_num )   

        
        
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        
        
        # image_list = list()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           hyper_params["epochs"])
                # image_list = unnormalize(val_images,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                # confusion_matrix.compute_matrix(val_lablist, pred_list, images=image_list, image_shape=(224,224,3))
                # ConfusionMatrixCallbackReuseImages(experiment, val_lablist, pred_list, confusion_matrix)
            
        val_accurate = acc / val_num
        
        # experiment.log_image(image_list,name=val_labels,image_shape=(224,224,3))
        # experiment.log_confusion_matrix(val_lebels_list, predict_y_list,images=image_list ,title="Confusion Matrix, Epoch #%d" %(epoch + 1),
        #                                 file_name="Confusion-Matrix-%03d.json"  %(epoch + 1),image_shape=(224,224,3))
        # 上傳到comet
        experiment.log_metrics({"accuracy": val_accurate, "loss": (running_loss / train_steps)}, epoch=epoch+1)
        
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_net = net
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    # log_model(experiment, net, model_name="mobilenet_v3_large")

        testdate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["val"])
        test_num = len(testdate_dataset)
        testdate_loader = torch.utils.data.DataLoader(testdate_dataset,
                                                    batch_size=hyper_params["batch_size"], shuffle=False,                                  
                                                    num_workers=nw)
        
        # UnNormalize tensor ToPILImage
        reback_img = transforms.Compose([
                    transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225])]
                    )

        best_net.eval()
        test_lebels_list = list()
        predict_y_list = list()
        acc = 0.0
        with torch.no_grad():
            confusion_matrix = experiment.create_confusion_matrix(labels=list(cla_dict.values()),max_categories=27,max_examples_per_cell=1000,image_shape=(224,224,3))
            test_bar = tqdm(testdate_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, hyper_params["epochs"])
                test_lebels_list += test_labels
                predict_y_list += predict_y

                # image_list = unnormalize(test_images,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                test_images = reback_img(test_images)

                # to pil_image
                image_list = list()
                for image in test_images:
                    image_list.append(transforms.ToPILImage()(image))

                confusion_matrix.compute_matrix(test_labels, predict_y, images=image_list, image_shape=(224,224,3))
                ConfusionMatrixCallbackReuseImages(experiment, test_labels, predict_y, confusion_matrix)
        
            experiment.log_confusion_matrix(
            matrix=confusion_matrix,
            title="Confusion Matrix, Epoch %d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
            )
            test_accurate = acc / test_num
        experiment.log_metrics({"test_accuracy": test_accurate}, epoch= epoch + 1)

    print('Finished Training')




if __name__ == '__main__':
    main()