# 字符检测系统-后端

[字符检测系统-前端](https://github.com/khuqen/letters_detect_frontend)

## 概述

字符识别系统(可检测A、B、C、D、X)五类目标，用于试卷的选择题部分的修改，此项目为后端。

使用pytorch构建目标检测模型：

- 数据集来自生成、学生手写字母和考研真实试卷，共467张图片
- 目标检测模型为Faster Rcnn模型
- 使用Flask部署模型并搭建后端
- 排序目标，返回排序的字母目标



相关说明：

1. `train_model/train_fasterrcnn.ipynb`，用于训练和测试模型，`train_model/dataset`为训练数据集
2. `app.py`，为基于flask搭建的检测接口和前后端交互接口
3. `net.py`，为封装的目标检测模型网络，提供检测、排序等功能



## 待完成

1. 数据库搭建
2. 用户系统搭建

