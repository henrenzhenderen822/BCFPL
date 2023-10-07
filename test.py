'''此程序用来测试，得到结果图像'''

import torch
import os
import cv2
import numpy as np
import time
from parameters import Parameters
from net.net_2conv import Binarynet
#
# 加载模型
modelname = 'checkpoints/MyNet/10-23_2134.pth'  # 已保存的模型文件
model = Binarynet().to('cuda')
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型

x = torch.rand(1, 3, 7, 7)

