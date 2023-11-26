import json
import os.path

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import GoogLeNet



def main():
    writer = SummaryWriter('logs')
    # 1、确定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
    print('Using device:{}'.format(device))

    # 2、确定需要执行的预处理
    data_transform = {
        'train':transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        'val':transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    }

    # 3、加载训练数据集和验证数据集，以及长度
    data_root = r'E:\研\研究生\code\AlexNet\flower_data'
    assert os.path.exists(data_root),"file path {} is not exist!".format(data_root)
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root,'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root,'val'),
                                       transform=data_transform['val'])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("train data have : {},val data have : {}".format(train_num,val_num))

    # 4、获取数据集中的类别，并写入json文件
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in flower_list.items())
    json_str = json.dumps(cla_dict,indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)

    # 5、加载训练数据集的loader和验证数据集的loader
    batch_size = 32
    nw = min([os.cpu_count(),batch_size if batch_size > 1 else 0,8])
    print("using num_workers is : {}".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers = nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = 32,
                                        shuffle = True,
                                        num_workers = nw)
    # 6、加载模型
    net = GoogLeNet(num_classes = 5,aux_logits = True,init_weights=True)

    # 7、设置损失函数和优化器
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0003)

    epochs = 30
    best_acc = 0.0
    save_path = './googLeNet.pth'
    for epoch in range(epochs):
        # 8、训练数据
        net.train()
        running_loss = 0.0
        for data in train_loader:
            images,labels = data
            # ①梯度清零
            optimizer.zero_grad()
            # ②图像预测
            logits,aux_logits2,aux_logits1 = net(images.to(device))
            # ③计算损失
            loss0 = loss_function(logits,labels.to(device))
            loss1 = loss_function(aux_logits1,labels.to(device))
            loss2 = loss_function(aux_logits2,labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            # ④传反向播
            loss.backward()
            # ⑤更新参数
            optimizer.step()
            # ⑥计算每一个epoch总损失
            running_loss += loss.item()
        writer.add_scalar('train_loss',running_loss,epoch+1)
        print("epoch : {},running_loss : {}".format(epoch+1,running_loss))

        net.eval()
        acc = 0.0
        with torch.no_grad():
            for data in val_loader:
                images,labels = data
                # ①预测图像
                outputs = net(images.to(device))
                # ②图像类别下标
                predict_y = torch.max(outputs,dim=1)[1]
                # print(predict_y) #输出该epoch中每张图片的类别的index
                # tensor([1, 3, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 4, 0, 4, 3, 3, 1, 1, 3, 4, 4, 0, 0,
                #         4, 3, 3, 4, 0, 4, 0, 3], device='cuda:0')
                # ③每个epoch的准确率
                acc += torch.eq(predict_y,labels.to(device)).sum().item()

        val_acc = acc / val_num
        print("epoch: {},val_AccRate:{}".format(epoch+1,val_acc))
        writer.add_scalar('val_AccRate',val_acc,epoch+1)
        if val_acc > best_acc:
            # 如果准确率超过了目前最好的准确率，则将当前准确率作为最好的准确率，并且保存模型
            best_acc = val_acc
            torch.save(net.state_dict(),save_path)
    print("Finished Running")
    writer.close()




if __name__ == '__main__':
    main()