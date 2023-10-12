import os

current_directory = os.getcwd()  # 获取当前路径
parent_directory = os.path.dirname(current_directory)  # 获取上一层路径

print("当前路径:", current_directory)
print("上一层路径:", parent_directory)
