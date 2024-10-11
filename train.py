from model import efficientnetv2_s as create_model
import os
import math
import argparse
import json
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
# from crop_dataset import MyDataSet

from miniimagenet import MiniImageNetDataset
from crop_dataset import MyDataSet
from loss import LDAMLoss,FocalLoss
from utils import read_split_data, train_one_epoch, evaluate
import csv
import codecs
import numpy as np
import matplotlib.pyplot as plt

class CustomCallback:
    def __init__(self, save_every=100):
        self.epoch_acc = []
        self.epoch_weights = []
        self.save_every = save_every


    def on_epoch_end(self, model, epoch, accuracy):
        self.epoch_acc.append(accuracy)
        if epoch % self.save_every == 0:
            weights = [param.detach().cpu().numpy().flatten() for param in model.parameters() if len(param.size()) > 1][:1]  # 仅保存前两层的权重
            self.epoch_weights.append(weights)
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.num_classes=num_classes
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def getClsNum(self):
        clsNum=[]
        for i in range(self.num_classes):
            clsNum.append(self.matrix[i,i])
        return clsNum

    def clear(self):
        self.matrix=np.zeros((self.num_classes, self.num_classes))

'''def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")'''

def curriculum_p(p, epoch):
    gamma = 0.0001
    return 1- (1.-p)*np.exp(-gamma*epoch)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    history = []
    start_epoch = -1
    p = 0.2

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # tb_writer = SummaryWriter()
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    if args.dataset == 'my_dataset':

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         #transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.594, 0.737,  0.956 ], [0.798, 0.816, 0.812])]),
            "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.594, 0.737,  0.956 ], [0.798, 0.816, 0.812])])}

        # # 实例化训练数据集
        # train_dataset = MyDataSet(images_path=train_images_path,
        #                           images_class=train_images_label,
        #                           transform=data_transform["train"])
        #
        # # 实例化验证数据集
        # val_dataset = MyDataSet(images_path=val_images_path,
        #                         images_class=val_images_label,
        #                         transform=data_transform["val"])
        train_dataset = MyDataSet(args.image_path, args.image_path + '/train.txt', data_transform["train"])

        train_num = len(train_dataset)
        cls_num_list = [0] * args.num_classes
        for label in train_dataset.targets:
            cls_num_list[label] += 1

            # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
        flower_list = {}
        dir_lst1 = os.listdir(os.path.join(args.image_path, "train"))
        for i in dir_lst1:
            flower_list[i] = dir_lst1.index(i)
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
        labels = [label for _, label in cla_dict.items()]
        validate_dataset = MyDataSet(args.image_path, args.image_path + '/val.txt', data_transform["val"])
    elif args.dataset == 'miniimagenet':
        train_transform = transforms.Compose([
            transforms.Resize(84),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = MiniImageNetDataset(root_dir=r'/media/kin/新加卷/mini-imagenet(1)/mini-imagenet/image_CDD/', split='train', transform1=train_transform)
        train_num = len(train_dataset)

        cls_num_list = [0] * args.num_classes
        for label in train_dataset.targets:
            cls_num_list[label] += 1

        flower_list = {}
        root_dir = r'/media/kin/新加卷/mini-imagenet(1)/mini-imagenet/image_CDD/'
        dir_lst1 = os.listdir(os.path.join(root_dir, "train"))
        for i in dir_lst1:
            flower_list[i] = dir_lst1.index(i)
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('imagenet_class_index.json', 'w') as json_file:
            json_file.write(json_str)
        labels = [label for _, label in cla_dict.items()]
        validate_dataset = MiniImageNetDataset(root_dir=r'/media/kin/新加卷/mini-imagenet(1)/mini-imagenet/image_CDD/', split='val',
                                           transform1=val_transform)



    #cls_num_list = [0] * args.num_classes
    #for label in train_dataset.targets:
        #cls_num_list[label] += 1

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    #flower_list = {}
    #dir_lst1 = os.listdir(os.path.join(args.image_path, "train"))
    #for i in dir_lst1:
       # flower_list[i] = dir_lst1.index(i)
    #cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    #json_str = json.dumps(cla_dict, indent=4)
   # with open('class_indices.json', 'w') as json_file:
        #json_file.write(json_str)

    #labels = [label for _, label in cla_dict.items()]
    # confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)

    batch_size = args.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    #validate_dataset = MyDataSet(args.image_path, args.image_path + '/val.txt', data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn)
    #
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)


    # 如果存在预训练权重则载入
    net = create_model(num_classes=args.num_classes).to(device)
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    print(net)
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))
    #
    # # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)
    if args.RESUME:
        # construct an optimizer
        # net.to(device)
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1E-4)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        path_checkpoint = "weights/lastModel.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        history=checkpoint['history']
        history = history.tolist()
        p = checkpoint['P']
        # p=checkpoint['P']
        # define loss function
        loss_function = nn.CrossEntropyLoss()
        # 是否冻结权重
        if args.freeze_layers:
            for name, para in net.named_parameters():
                # 除最后的全连接层外，其他权重全部冻结
                if "Drop1" in name:
                    para.requires_grad_(True)
                elif "FC1" in name:
                    para.requires_grad_(True)
                else:
                    para.requires_grad_(False)
    else:
        # net.to(device)
        # 是否冻结权重
        if args.freeze_layers:
            for name, para in net.named_parameters():
                # 除最后的全连接层外，其他权重全部冻结
                if "Drop1" in name:
                    para.requires_grad_(True)
                elif "FC1" in name:
                    para.requires_grad_(True)
                else:
                    para.requires_grad_(False)
        # define loss function
        # loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        # loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        #lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    callback = CustomCallback()

    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    # cvsPath = r"E:\zh\analyseData\2paper\resnet50_37classAguBalance_batchsize8.csv"
    for epoch in range(args.epochs):

        # train
        if epoch < start_epoch:
            continue
        # train
        net.train()
        # p = curriculum_p(p, epoch)
        # if p > 0.5:
        #     p = 0.5


        if args.DRW:
            if epoch<=30:
                betas = 0.9999
                effective_num = 1.0 - np.power(betas, cls_num_list)
                per_cls_weights = (1.0 - betas) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            else:
                betas = 0.9999
                effective_num = 1.0 - np.power(betas, cls_num_list)
                per_cls_weights = (1.0 - betas) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

                clsNum = confusion.getClsNum()
                effective_num1 = 1.0 - np.power(betas, clsNum)
                per_cls_weights1 = (1.0 - betas) / np.array(effective_num1)
                per_cls_weights1 = per_cls_weights1 / np.sum(per_cls_weights1) * len(clsNum)
                per_cls_weights1 = torch.FloatTensor(per_cls_weights1).to(device)

                per_cls_weights = per_cls_weights * per_cls_weights1
            # if epoch<80:
            #     betas = 0
            #     effective_num = 1.0 - np.power(betas, cls_num_list)
            #     per_cls_weights = (1.0 - betas) / np.array(effective_num)
            #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            #     per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            # elif epoch<160:
            #     betas = 0.9999
            #     effective_num = 1.0 - np.power(betas, cls_num_list)
            #     per_cls_weights = (1.0 - betas) / np.array(effective_num)
            #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            #     per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            # else:
            #     betas = 0.9999
            #     effective_num = 1.0 - np.power(betas, cls_num_list)
            #     per_cls_weights = (1.0 - betas) / np.array(effective_num)
            #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            #     per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
            #
            #     clsNum = confusion.getClsNum()
            #     effective_num1 = 1.0 - np.power(betas, clsNum)
            #     per_cls_weights1 = (1.0 - betas) / np.array(effective_num1)
            #     per_cls_weights1 = per_cls_weights1 / np.sum(per_cls_weights1) * len(clsNum)
            #     per_cls_weights1 = torch.FloatTensor(per_cls_weights1).to(device)
            #
            #     per_cls_weights = per_cls_weights * per_cls_weights1

        confusion.clear()
        if args.loss_type == 'Focal':
            loss_function=FocalLoss(weight=per_cls_weights, gamma=1)
        elif args.loss_type == 'LDAM':
            loss_function=LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
        elif args.loss_type == 'our':
            loss_function = nn.CrossEntropyLoss(weight=per_cls_weights)
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        acc1 = 0.0
        for step, data in enumerate(train_bar):
            images1,images2,images3, labels = data
            images1 = images1.to(device)
            images2 = images2.to(device)
            images3 = images3.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = net((images1,images2,images3))
            train_y = torch.max(logits, dim=1)[1]
            confusion.update(train_y.to("cpu").numpy(), labels.to("cpu").numpy())
            acc1 += torch.eq(train_y, labels).sum().item()
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

        #scheduler.step()

        net.eval()
        val_loss = 0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images1, val_images2, val_images3, val_labels = val_data
                val_images1 = val_images1.to(device)
                val_images2 = val_images2.to(device)
                val_images3 = val_images3.to(device)
                val_labels = val_labels.to(device)
                outputs = net((val_images1,val_images2,val_images3))
                loss = loss_function(outputs, val_labels)
                val_loss += loss.item()
                # valid_losses.append(loss.item())
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        val_accurate = acc / val_num
        train_accurate = acc1 / train_num
        valid_loss = val_loss / val_steps
        history.append([train_accurate, running_loss / train_steps, val_accurate, valid_loss])
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d] train_accuracy: %.4f train_loss: %.4f  val_accuracy: %.4f val_loss: %.4f' %
              (epoch + 1, train_accurate, running_loss / train_steps, val_accurate, valid_loss))
        if not os.path.isdir("./weights_mini_5_1"):
            os.mkdir("./weights_mini_5_1")
        mind_history = history.copy()
        mind_history = np.array(mind_history)

        callback.on_epoch_end(net, epoch, val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            # 模型保存
            checkpoint = {
                'epoch': epoch + 1,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': mind_history,
                # 'P': p
                # 'scheduler':scheduler
                # 'P': p
            }
            torch.save(checkpoint, "./weights_mini_5_1/bestModel.pth")
        # 模型保存
        checkpoint = {
            'epoch': epoch + 1,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': mind_history,
            # 'P': p
            # 'scheduler':scheduler
            # 'P': p
        }
        torch.save(checkpoint, "./weights_mini_5_1/lastModel.pth")
        '''data_write_csv(args.cvsPath, mind_history)'''


    print('Finished Training')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(callback.epoch_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
        # tags = ["loss", "accuracy", "learning_rate"]
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        #
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)#100 or 37
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--DRW', type=bool, default=True)
    parser.add_argument('--loss_type', type=str, default="our")

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--image_path', type=str,
                        default=r"/media/kin/新加卷/mini-imagenet(1)/mini-imagenet/image_CDD/")
    parser.add_argument('--cvsPath', type=str,
                        default=r"/media/kin/新加卷/小论文实验汇总/B-CRSDD-LT/LCFF_efficientnetv2/LCFF_efficientnetV2-Agudata-batchsize32/32_mini_5:1.csv")

    # shufflenetv2_x1.0 官方权重下载地
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dataset', default='miniimagenet', choices=['my_dataset', 'miniimagenet'])
    opt = parser.parse_args()

    main(opt)


