'''此程序用来配置参数和网络模型'''
# 实验需要经常更改参数，利用此程序布局配置
import argparse
import sys


config = {
    'batch_size': 128,
    'img_size': 9,
    'input_size': 50,
    'epochs': 20,
    'learning_rate': 0.001,
    'dataset': 'CNRPark',
    'model': 'trained_model.pth',
    'park_img': 'parking.jpg',
    'park_info': 'parking.csv'
}

parser = argparse.ArgumentParser(description="demo of argparse")
# 通过对象的add_argument函数来增加参数。
parser.add_argument('-B', '--batch_size', default=config["batch_size"], type=int)
parser.add_argument('-I', '--img_size', default=config["img_size"], type=int)
parser.add_argument('--input_size', default=config["input_size"], type=int)
parser.add_argument('-E', '--epochs', default=config["epochs"], type=int)
parser.add_argument('-L', '--learning_rate', default=config["learning_rate"], type=float)
parser.add_argument('-D', '--dataset', default=config["dataset"], type=str, help="Choose dataset")
parser.add_argument('-M', '--model', default=config["model"], type=str)
parser.add_argument('-P', '--park_img', default=config["park_img"], type=str, help="Complete parking image")
parser.add_argument('-C', '--park_info', default=config["park_info"], type=str, help="Parking space coordinate")
args = parser.parse_args()
size = args.img_size


'''Parameters类保存运行所需要的参数等配置(类属性，可以直接通过类名调用)'''
class Parameters():
    batch_size = args.batch_size         # 批大小
    img_size = args.img_size             # 图片大小
    input_size = args.input_size         # 输入大小
    epochs = args.epochs                 # 跑epochs轮数据集
    learning_rate = args.learning_rate   # 学习率大小
    dataset = args.dataset               # 数据集


def main():
    print(args)
    # vars() 函数返回对象object的属性和属性值的字典对象。
    ap = vars(args)
    print(ap)
    print(ap['batch_size'])  # Li
    print('Hello {} {}'.format(Parameters.batch_size, Parameters.img_size))  # Hello Li 21

    print("Input argument is %s" %(sys.argv))


if __name__ == '__main__':
    main()
