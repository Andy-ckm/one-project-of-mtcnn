import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tools import utils
import nets as nets
from torchvision import transforms
import time
import os


#   网络调参
#   p网络
p_cls = 0.6     # 0.6
p_nms = 0.5     # 0.5
#   R网络
r_cls = 0.6     # 0.6
r_nms = 0.5     # 0.5
#   O网络
o_cls = 0.97    # 0.3
o_nms = 0.7     # 0.5

#   侦测器
class Detector(object):
    #   初始化时加载三个网络权重，cuda默认打开
    def __init__(self, pnet_param=r"./param_0/pnet.pt", rnet_param=r"./param_0/rnet.pt",
                 onet_param=r"./param_0/onet.pt", isCuda=True):

        self.isCuda = isCuda
        #   创建实例变量，实例化网络
        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()
        #   加入cuda加速训练
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        #   把训练好的权重加载到p网络中
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(onet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        #   调用测试模式，因为网络中含有BN
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        #   图片数据类型转换
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    #   p网络的检测
    def detect(self, image):
        #   p网络检测----1st
        #   开始计时
        start_time = time.time()
        #   调用__pnet_detect_函数
        pnet_boxes = self.__pnet_detect(image)
        #   排除框数量为空的异常情况
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        #   r网络检测----2nd
        start_time = time.time()
        #   传入原图，P网络的一些框，根据这些框在原图上抠图
        rnet_boxes = self.__image_transform(image)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        # return rnet_boxes

        #   o网络检测----3rd
        start_time = time.time()
        onet_boxes = self.__image_transform(image)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()

    #   创建p网络检测器
    def __pnet_detect(self, image):
        #   p网络全部是卷积(FCN)，与输入图片大小无关
        #   创建空列表，接收符合条件的建议框
        boxes = []
        img = image
        w, h = img.size
        #   获取图片的最小边长
        min_side_len = min(w, h)
        #   初始缩放比例（为1时则不缩放），可得到不同的图片
        scale = 1
        #   直到缩放小于12时停止
        while min_side_len > 12:
            #   将图片数组转为张量
            img_data = self.__image_transform(img)
            if self.isCuda:
                #   将Tensor形式的图片传到cuda里加速
                img_data = img_data.cuda()
            #   并在“批次"上升维(测试时传的不止一张图片)
            img_data.unsqueeze_(0)
            # print("img_data:", img_data.shape)  # [1, 3, 416, 500]:c=3,w=416,h=500
            #   返回多个置信度和偏移量
            _cls, _offset = self.pnet(img_data)
            # print("_cls", _cls)     # [1, 1, 203, 245]
            # print("_offset", _offset)   # [1, 4, 203, 245]
            cls = _cls[0][0].cpu().data     # [203,245]:分组卷积特征图的尺寸:w,h
            offset = _offset[0].cpu().data  # [4,203,245] 分组卷积的通道尺寸：c,w,h
            #   置信度大于0.6的框索引
            idxs = torch.nonzero(torch.gt(cls, p_cls))
            #   根据索引，依次添加符合条件的框：ds[idx[0],idx[1]],在置信度中取值：idx[0]行索引，idx[1]列索引
            for idx in idxs:
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], scale))
            #   调用框反算函数_box,把大于0.6的框留下
            scale *= 0.7    # 缩放图片：循环控制条件
            #   新的宽度和高度
            _w = int(w * scale)
            _h = int(h * scale)
            #   根据缩放后的宽和高，对图片进行缩放
            img = img.resize()
            #   重新获取最小的宽和高
            min_side_len = min(_w, _h)
            return utils.nms(np.array(boxes), p_nms)

    #   特征反算：将回归量还原到原图上去，根据特征图反算到原图的建议框
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        #   p网络池化步长为2
        #   特征反算时“行索引，索引互换”，原为[0]
        _x1 = (start_index[1].float() * stride) / scale
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len) / scale
        _y2 = (start_index[0].float() * stride + side_len) / scale
        #   人脸所在区域建议框的宽和高
        ow = _x2 - _x1
        oh = _y2 - _y1
        #   根据idxs行索引和列索引，找到对应偏移量.[x1,y1,x2,y2]
        _offset = offset[:, start_index[0], start_index[1]]
        #   根据偏移量算实际框的位置
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        #   正式框：返回四个坐标点和一个偏移量
        return [x1, y1, x2, y2]

    #   创建R网络检测函数
    def __rnet_detect(self, image, pnet_boxes):
        #   创建空列表，存放抠图
        _img_dataset = []
        #   给p网络输出的框找出中心点，沿着最大边长的两边扩充成正方形再抠图
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        #   遍历每个框，每个框返回框四个坐标点，抠图，缩放。数据类型转换，添加列表
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            #   根据四个点的坐标抠图
            img = image.crop((_x1, _x2, _y1, _y2))
            #   缩放固定尺寸
            img = img.resize((124, 124))
            #   将图片数组转化为张量
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)
        #   stack堆叠(默认在0轴)，此处相当数据类型转换
        img_dataset = torch.stack(_img_dataset)
        #   加入cuda计算
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        #   将24 * 24 的图片传入网络再进行一次筛选
        _cls, _offset = self.rnet(img_dataset)
        #   将gpu上的数据放在cpu上去，再转成数组numpy
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        # print("r_cls:", cls.shape)      # (11,1):p网络生成了11个框
        # print("r_offset", offset)       # (11,4)
        boxes = []  # R网络要留下来的框，存到boxes里面
        idxs, _ = np.where(cls > r_cls)     # 原置信度0.6是偏低的
        #   根据索引，遍历符合条件的框；1轴上的索引恰为符合条件的置信度索引
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            #   基准框的宽
            ow = _x2 - _x1
            oh = _y2 - _y1
            #   实际框的坐标点
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            #   返回4个坐标点和置信度
            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        #   原r_nms为0.5
        return utils.nms(np.array(boxes), r_nms)
