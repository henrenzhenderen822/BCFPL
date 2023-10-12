'''此程序用来测试，得到结果图像'''

import torch
import os
import cv2
import numpy as np
import time
from parameters import Parameters
from model import Binarynet
#
# 加载模型
modelname = '../checkpoints/10-09_2208.pth'  # 已保存的模型文件
model = Binarynet().to('cuda')
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型

# 加载图像
img_name = 'parking.jpg'
image = cv2.imread(img_name)

# 加载停车位的坐标数据
filename = 'parking.csv'

'''★★★★★在此修改单个停车位的图片大小★★★★★'''
img_size = Parameters.img_size
input_size = Parameters.input_size
# print(patch_size)


# 将opencv图片格式转化为torch格式
def make_suit_torch(img):
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).div(255.0).unsqueeze(0)  # 除255
    img = img.permute(0, 3, 1, 2)
    img = img.float()

    # print(img)
    return img


# 将图片分割成小块并送入二分类网络检测
def cut_image(image, filename, model):

    points = np.genfromtxt(filename, delimiter=',')
    for point in points[1:]:
        (x1, y1, x2, y2) = point[1:]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        patch = image[y1:y2, x1:x2]
        patch = make_suit_torch(patch)  # 转化为适合pytorch的数据(大小要对应神经网络的输入大小)

        patch = patch.to('cuda')

        # 这里必须加 model.eval() 因为测试单张图像要固定住BatchNorm层
        model.eval()
        with torch.no_grad():
            logits = model(patch)           # 带入二分类模型

        pred = logits.argmax(dim=1)

        if pred == 1:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画出矩形
        if pred == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 225, 0), 2)  # 画出矩形
    return image


# 此函数用来展示图片
def cv_show(img, name='test'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 展示结果图
image = cut_image(image, filename, model)
cv_show(image)

# 以日期为名称保存结果图片
now = time.strftime('%m-%d_%H%M_%S')
imgname = now + '.jpg'
cv2.imwrite(imgname, image)



