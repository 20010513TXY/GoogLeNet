import json
import os.path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model import GoogLeNet


def main():
    # ①确定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
    # ②确定数据预处理
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    # ③导入待预测图像
    img_path = 'E:\研\研究生\code\AlexNet\向日葵.jpg'
    assert os.path.exists(img_path),'file path {} is not exist!'.format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    # ④加载class_indices.json文件
    json_path = r'class_indices.json'
    with open(json_path,'r') as f:
        class_indict = json.load(f)

    # ⑤加载模型
    model = GoogLeNet(num_classes= 5,aux_logits=False)
    model.to(device)

    # ⑥加载权重
    weights_path = r'googLeNet.pth'
    assert os.path.exists(weights_path),'file path {} is not exist!'.format(weights_path)
    model.load_state_dict(torch.load(weights_path,map_location=device),strict=False)

    # ⑦图像预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # ⑧绘图
    plt.title('class:{} prob:{:5}'.format(class_indict[str(predict_cla)],
                                        predict[predict_cla].numpy()))
    plt.show()
    for i in range(len(predict)):
        print('class:{} prob:{}'.format(class_indict[str(i)],predict[i].numpy()))

if __name__ == '__main__':
    main()