import os
import sys
import json

from comet_ml import Experiment
from torch import Tensor
from typing import List
from torchsampler import ImbalancedDatasetSampler
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm


from model_v3 import mobilenet_v3_large
from comet_ml import ConfusionMatrix
from comet_ml.integration.pytorch import log_model


experiment = Experiment(
  api_key = "Lci5cxkzmbkt3kFnJf1tXlNE1",
  project_name = "comet-test",
  workspace="karasvii",
  display_summary_level=0,
)


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


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def unnormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    _assert_image_tensor(tensor)

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.mul_(mean).add_(std)
        


def main():
    device = "mps" if getattr(torch,'has_mps',False) \
    else "gpu" if torch.cuda.is_available() else "cpu"
    print("using {} device.".format(device))




    experiment.set_name("MobileNetV3-16-10")
    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 16,
    "num_classes": 27,
    }
    experiment.log_parameters(hyper_params)



# ToTensor H，W，C ——> C，H，W C/255 [0~1]
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "code ref/Test6_mobilenet/data_set", "flower_data")  # flower data set path
    
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), hyper_params["batch_size"] if hyper_params["batch_size"] > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            sampler=ImbalancedDatasetSampler(train_dataset),
                                            batch_size=hyper_params["batch_size"],
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
    net = mobilenet_v3_large(num_classes=hyper_params["num_classes"])

    # Initialize and train your model
    # model = TheModelClass()
    # train(model)

    # Seamlessly log your Pytorch model
    log_model(experiment, net, model_name="mobilenet_v3_large")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v3_large.pth"
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
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=hyper_params["learning_rate"])
    best_net = mobilenet_v3_large(num_classes=hyper_params["num_classes"])
    best_acc = 0.0
    save_path = './MobileNetV3.pth'
    train_steps = len(train_loader)
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
        image_list = list()
        with torch.no_grad():
            confusion_matrix = experiment.create_confusion_matrix()
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           hyper_params["epochs"])
                image_list = unnormalize(val_images,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                confusion_matrix.compute_matrix(val_labels, predict_y, images=image_list, image_shape=(224,224,3))
                ConfusionMatrixCallbackReuseImages(experiment, val_labels, predict_y, confusion_matrix)
                

            experiment.log_confusion_matrix(
            matrix=confusion_matrix,
            title="Confusion Matrix, Epoch #0",
            file_name="confusion-matrix-%03d.json" % 0,
            ); 
            
        val_accurate = acc / val_num
        
        # experiment.log_image(image_list,name=val_labels,image_shape=(224,224,3))
        # experiment.log_confusion_matrix(val_lebels_list, predict_y_list,images=image_list ,title="Confusion Matrix, Epoch #%d" %(epoch + 1),
        #                                 file_name="Confusion-Matrix-%03d.json"  %(epoch + 1),image_shape=(224,224,3))
        # 上傳到comit
        experiment.log_metrics({"accuracy": val_accurate, "loss": (running_loss / train_steps)}, epoch=epoch+1)
        
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_net = net
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    # log_model(experiment, net, model_name="mobilenet_v3_large")
        print('Finished Training')



if __name__ == '__main__':
    main()
