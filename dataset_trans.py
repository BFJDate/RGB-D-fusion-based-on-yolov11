import os
import cv2 as cv
import pandas as pd
import ast

# csv2yolo
# trainset
trainset_Path = "D:/Code/yolov11/ultralytics-main/data/SPARK-ICIP-2021/ICIP-2021/train_labels.csv"
trainset = pd.read_csv(trainset_Path)
# 数据处理，添加txt后缀列方便后续处理，同时添加列将class转换为label方便后续输出
trainset['rgbtxt'] = trainset['image']
trainset['rgbtxt'] = trainset['rgbtxt'].str.replace("png","txt")
trainset['depthtxt'] = trainset['depth']
trainset['depthtxt'] = trainset['depthtxt'].str.replace("png","txt")
trainset['label'] = trainset['class'].replace({"AcrimSat":"0","Aquarius":"1","Aura":"2","Calipso":"3","Cloudsat":"4","CubeSat":"5","Debris":"6","Jason":"7","Sentinel-6":"8","Terra":"9","TRMM":"10"})
#将csv数据集中xyxy格式坐标转换为yolo数据集中xyhw坐标
trainset['bbox'] = trainset['bbox'].apply(ast.literal_eval)
trainset['bbox'] = trainset['bbox'].apply(lambda bbox: [
    ((bbox[3] + bbox[1]) / 2) / 1024,   # 新的 x1（中心点 x 坐标）
    ((bbox[2] + bbox[0]) / 2) / 1024,   # 新的 y1（中心点 y 坐标）
    (bbox[3] - bbox[1]) / 1024,     # 宽度（x2 - x1）
    (bbox[2] - bbox[0]) / 1024      # 高度（y2 - y1）

])
# 写入文件
trainrgblabel_name = trainset.iloc[:,6]
traindepthlabel_name = trainset.iloc[:,7]
train_class = trainset.iloc[:,0]
train_label = trainset.iloc[:,8]
train_bbox = trainset.iloc[:,4]
for i, name in enumerate(trainrgblabel_name):
    path = os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\RGB\train\labels", train_class[i])
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\RGB\train\labels", train_class[i], name), 'w') as fp:
        fp.write(train_label[i])
        fp.write(" ")
        fp.write(' '.join(map(str, train_bbox[i])))
for i, name in enumerate(traindepthlabel_name):
    path = os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\depth\train\labels", train_class[i])
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\depth\train\labels", train_class[i], name), 'w') as fp:
        fp.write(train_label[i])
        fp.write(" ")
        fp.write(' '.join(map(str, train_bbox[i])))
        
# testset
testset_Path = "D:/Code/yolov11/ultralytics-main/data/SPARK-ICIP-2021/ICIP-2021/test_labels.csv"
testset = pd.read_csv(testset_Path)
# 数据处理，添加txt后缀列方便后续处理，同时添加列将class转换为label方便后续输出
testset['rgbtxt'] = testset['image']
testset['rgbtxt'] = testset['rgbtxt'].str.replace("png","txt")
testset['depthtxt'] = testset['depth']
testset['depthtxt'] = testset['depthtxt'].str.replace("png","txt")
#将csv数据集中xyxy格式坐标转换为yolo数据集中xyhw坐标
testset['bbox'] = testset['bbox'].apply(ast.literal_eval)
testset['bbox'] = testset['bbox'].apply(lambda bbox: [
    ((bbox[3] + bbox[1]) / 2) / 1024,
    ((bbox[2] + bbox[0]) / 2) / 1024,   
    (bbox[3] - bbox[1]) / 1024,
    (bbox[2] - bbox[0]) / 1024                
])
# 写入文件
testrgblabel_name = testset.iloc[:,5]
testdepthlabel_name = testset.iloc[:,6]
test_label = testset.iloc[:,3]
test_bbox = testset.iloc[:,4]
for i, name in enumerate(testrgblabel_name):
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\RGB\test\labels", name), 'w') as fp:
        fp.write(str(test_label[i]))
        fp.write(" ")
        fp.write(' '.join(map(str, test_bbox[i])))
for i, name in enumerate(testdepthlabel_name):
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\depth\test\labels", name), 'w') as fp:
        fp.write(str(test_label[i]))
        fp.write(" ")
        fp.write(' '.join(map(str, test_bbox[i])))

# csv2yolo
# valset
valset_Path = "D:/Code/yolov11/ultralytics-main/data/SPARK-ICIP-2021/ICIP-2021/validate_labels.csv"
valset = pd.read_csv(valset_Path)
# 数据处理，添加txt后缀列方便后续处理，同时添加列将class转换为label方便后续输出
valset['rgbtxt'] = valset['image']
valset['rgbtxt'] = valset['rgbtxt'].str.replace("png","txt")
valset['depthtxt'] = valset['depth']
valset['depthtxt'] = valset['depthtxt'].str.replace("png","txt")
valset['label'] = valset['class'].replace({"AcrimSat":"0","Aquarius":"1","Aura":"2","Calipso":"3","Cloudsat":"4","CubeSat":"5","Debris":"6","Jason":"7","Sentinel-6":"8","Terra":"9","TRMM":"10"})
#将csv数据集中xyxy格式坐标转换为yolo数据集中xyhw坐标
valset['bbox'] = valset['bbox'].apply(ast.literal_eval)
valset['bbox'] = valset['bbox'].apply(lambda bbox: [
    ((bbox[3] + bbox[1]) / 2) / 1024, 
    ((bbox[2] + bbox[0]) / 2) / 1024,  
    (bbox[3] - bbox[1]) / 1024,
    (bbox[2] - bbox[0]) / 1024    
])
# 写入文件
valrgblabel_name = valset.iloc[:,6]
valdepthlabel_name = valset.iloc[:,7]
val_class = valset.iloc[:,0]
val_label = valset.iloc[:,8]
val_bbox = valset.iloc[:,4]
for i, name in enumerate(valrgblabel_name):
    path = os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\RGB\validate\labels", val_class[i])
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\RGB\validate\labels", val_class[i], name), 'w') as fp:
        fp.write(val_label[i])
        fp.write(" ")
        fp.write(' '.join(map(str, val_bbox[i])))
for i, name in enumerate(valdepthlabel_name):
    path = os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\depth\validate\labels", val_class[i])
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(r"D:\Code\yolov11\ultralytics-main\data\SPARK-ICIP-2021\ICIP-2021\depth\validate\labels", val_class[i], name), 'w') as fp:
        fp.write(val_label[i])
        fp.write(" ")
        fp.write(' '.join(map(str, val_bbox[i])))