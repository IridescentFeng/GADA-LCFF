import os
import json
# from torchvision import transforms, datasets, utils
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import efficientnetv2_s as create_model
import csv
from crop_dataset import MyDataSet

# from model import MobileNetV2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        allTP=[]
        allPrecision=[]
        allRecall=[]
        allSpecificity=[]
        allF1=[]
        allAcc=[]
        sum_TP = 0
        matrix_csvFile = open("matrix.csv", "w", newline="")
        matrix_writer = csv.writer(matrix_csvFile)
        for i in range(num_classes):
            matrix_writer.writerow(self.matrix[i])
        matrix_csvFile.close()
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1=round(2*Precision*Recall/(Precision+Recall),3) if Precision+Recall != 0 else 0.
            acc=round((TP+TN)/(TP+TN+FP+FN),3) if TP+TN+FP+FN != 0 else 0.
            allAcc.append(acc)
            allF1.append(F1)
            allTP.append(TP)
            allPrecision.append(Precision)
            allRecall.append(Recall)
            allSpecificity.append(Specificity)
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return allTP,allPrecision,allRecall,allSpecificity,allAcc,allF1

    def plot(self):
        matrix = self.matrix

        # classNum={}
        # #获取类别对于的类别数的字典
        # for i in range(self.num_classes):
        #     sumNum=sum(matrix[:,i])
        #     classNum[str(i)]=int(sumNum)
        # #对类别字典进行排序，按类别数，升序排列
        # classNum_order = sorted(classNum.items(), key=lambda x: x[1], reverse=False)
        # print(classNum_order)
        # lowClass=[]
        # mindClass=[]
        # highClass=[]
        # #获取低五类混淆矩阵
        # for i in range(5):
        #     # print(classNum_order[i][0])
        #     lowClass.append(int(classNum_order[i][0]))
        #     mindClass.append(int(classNum_order[num_classes//2-2+i][0]))
        #     highClass.append(int(classNum_order[num_classes-1-i][0]))
        # print("lowClass:",lowClass)
        # print("mindClass:", mindClass)
        # print("highClass:", highClass)
        matrix1=[]
        matrix1=matrix.copy()
        for i in range(self.num_classes):
            matrix1[:,i]=matrix1[:,i]/sum(matrix1[:,i])
        # # print(matrix)
        # lowNumRecall=[]
        # print("数量最低5类的召回率：")
        # for i in lowClass:
        #     print("flaw"+str(i)+":", round(matrix1[i,i],3))
        #     lowNumRecall.append(round(matrix1[i,i],3))
        # mindNumRecall = []
        # print("数量居中5类的召回率：")
        # for i in mindClass:
        #     print("flaw" + str(i) + ":", round(matrix1[i, i], 3))
        #     mindNumRecall.append(round(matrix1[i, i], 3))
        # highNumRecall = []
        # print("数量最高5类的召回率：")
        # for i in highClass:
        #     print("flaw" + str(i) + ":", round(matrix1[i, i], 3))
        #     highNumRecall.append(round(matrix1[i, i], 3))
        #
        # allRecall = {}
        # # 获取所有类别的召回率
        # for i in range(self.num_classes):
        #     allRecall[str(i)] = round(matrix1[i, i], 3)
        # # 对召回率排序,升序
        # allRecall_order = sorted(allRecall.items(), key=lambda x: x[1], reverse=False)
        # print("allRecall_order:", allRecall_order)
        # lowRecallClass = []
        # mindRecallClass = []
        # highRecallClass = []
        # # 获取高中低各五类的类别
        # for i in range(5):
        #     lowRecallClass.append(int(allRecall_order[i][0]))
        #     mindRecallClass.append(int(allRecall_order[num_classes // 2 - 2 + i][0]))
        #     highRecallClass.append(int(allRecall_order[num_classes - 1 - i][0]))
        # print("lowRecallClass:", lowRecallClass)
        # print("mindRecallClass:", mindRecallClass)
        # print("highRecallClass:", highRecallClass)
        #
        # lowRecall = []
        # print("召回率最低的5类：")
        # for i in lowRecallClass:
        #     print("flaw" + str(i) + ":", round(matrix1[i, i], 3))
        #     lowRecall.append(round(matrix1[i, i], 3))
        # mindRecall = []
        # print("召回率居中的5类：")
        # for i in mindRecallClass:
        #     print("flaw" + str(i) + ":", round(matrix1[i, i], 3))
        #     mindRecall.append(round(matrix1[i, i], 3))
        # highRecall = []
        # print("召回率最高的5类：")
        # for i in highRecallClass:
        #     print("flaw" + str(i) + ":", round(matrix1[i, i], 3))
        #     highRecall.append(round(matrix1[i, i], 3))


        # lowMatrix = matrix.take(lowClass, axis=1)#最低五类的混淆矩阵
        # lowMatrix = lowMatrix.take(lowClass, axis=0)
        # lowMatrix=np.round(lowMatrix,2)
        #
        # mindMatrix = matrix.take(mindClass, axis=1)
        # mindMatrix = mindMatrix.take(mindClass, axis=0)
        # mindMatrix = np.round(mindMatrix, 2)
        #
        # highMatrix = matrix.take(highClass, axis=1)
        # highMatrix = highMatrix.take(highClass, axis=0)
        # highMatrix = np.round(highMatrix, 2)
        plt.imshow(matrix1, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45,fontsize=10)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels,fontsize=10)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels',fontsize=15)
        plt.ylabel('Predicted Labels',fontsize=15)
        plt.title('Confusion matrix',fontsize=15)

        # 在图中标注数量/概率信息
        # thresh = 0.5
        thresh = matrix1.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                # info = int(matrix[y, x])
                info=round(matrix1[y, x], 2)
                if info==0:
                    info=int(info)
                else:
                    info=info
                plt.text(x, y, info,fontsize=10,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # data_transform = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"
    data_transform=transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                        transforms.ToTensor(),
                        transforms.Normalize([0.594, 0.737,  0.956 ], [0.798, 0.816, 0.812])])

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = r"/media/kin/LENOVO_USB_HDD/AguBalance37/normal"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    test_dataset = MyDataSet(image_path, image_path + '/val.txt', data_transform)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform)
    num_classes=37
    batch_size = 32
    testdate_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    # net = MobileNetV2(num_classes=5)
    net = create_model(num_classes=num_classes).to(device)
    # load pretrain weights
    path_checkpoint = "weights/bestModel.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)
    p = 0.2

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
    # net.eval()
    # with torch.no_grad():
    #     for val_data in tqdm(testdate_loader):
    #         val_images, val_labels = val_data
    #         outputs = net(val_images.to(device))
    #         outputs = torch.softmax(outputs, dim=1)
    #         outputs = torch.argmax(outputs, dim=1)
    #         confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    # confusion.plot()
    # confusion.summary()
    # csvFile = open("TP-Precision-Recall-Specificity.csv", "w", newline="")
    # writer = csv.writer(csvFile)
    # writer.writerow(["TP", "Precision", "Recall", "Specificity", "Acc", "F1"])
    #
    # Precision = []
    # Recall = []
    # Specificity = []
    # Acc = []
    # F1 = []
    # labels = [label for _, label in class_indict.items()]
    # confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
    csvFile = open("TP-Precision-Recall-Specificity.csv", "w", newline="")
    writer = csv.writer(csvFile)
    writer.writerow(["TP", "Precision", "Recall", "Specificity", "Acc", "F1"])

    Precision = []
    Recall = []
    Specificity = []
    Acc = []
    F1 = []
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(testdate_loader):
            val_images1, val_images2, val_images3, val_labels = val_data
            val_images1 = val_images1.to(device)
            val_images2 = val_images2.to(device)
            val_images3 = val_images3.to(device)
            val_labels = val_labels.to(device)
            outputs = net((val_images1, val_images2, val_images3, p))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    # confusion.plot()
    TP, Precision, Recall, Specificity, Acc, F1 = confusion.summary()
    macro_Precision = sum(Precision) / len(Precision)
    macro_Recall = sum(Recall) / len(Recall)
    macro_acc = sum(Acc) / len(Acc)
    macro_F1 = sum(F1) / len(F1)
    print("macro_Precision:", macro_Precision)
    print("macro_Recall:", macro_Recall)
    # print("macro_acc:", macro_acc)
    print("macro_F1:", macro_F1)
    # print(Precision)
    # writer.writerow([Precision, Recall, Specificity])
    for i in range(num_classes):
        writer.writerow([TP[i], Precision[i], Recall[i], Specificity[i], Acc[i], F1[i]])
    csvFile.close()

