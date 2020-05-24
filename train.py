#   创建训练器


import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from sampling import FaceDataset


# 创建训练器
class Trainer(object):
    def __init__(self, net, save_path, dataset_path, isCuda = True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda
        # self.device = device

        #   判断是否有cuda
        if self.isCuda:
            self.net.cuda()
        #   第二种方式
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #   创建损失函数
        #   置信度损失函数：二分类交叉熵损失函数，是多分类交叉熵的一个特例（前面必须有sigmoid函数激活）
        self.cls_loss_fn = nn.BCELoss()
        #   偏移量损失
        self.offset_loss_fn = nn.MSELoss()
        #   创建优化器
        self.optimizer = optim.Adam(self.net.parameters())
        #   恢复网络训练（加载模型参数）
        if os.path.exists(self.save_path):
            #   如果文件存在，则继续训练
            net.load_state_dict(torch.load(self.save_path))

    #   训练方法
    def train(self):
        #   加载数据
        faceDataset = FaceDataset(self.dataset_path)
        #   数据加载器
        #   num_workers=4：有4个线程在加载数据(加载数据需要时间，以防空置)；drop_last：为True时表示，防止批次不足报错。
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=4,
                                drop_last=True)

        while True:
            #   枚举样本，置信度，偏移量
            for i, (img_data_, category_, offset_) in enumerate(dataloader):

                #   判断是否为CUDA环境
                if self.isCuda:
                    img_data_ = img_data_.cuda()  # [512,3,12,12]
                    category_ = category_.cuda()  # [512,1]
                    offset_ = offset_.cuda()  # [512,4]

                #   网络输出
                _output_category, _output_offset = self.net(img_data_)
                # print(_output_category.shape)  # [512,1,1,1]
                # print(_output_offset.shape)  # [512,4,1,1]
                output_catagory = _output_category.view(-1, 1)  # [512,1]
                output_offset = _output_offset.view(-1, 4)  # [512,4]
                # output_landmark = _output_landmark.view(-1, 10)   # [512, 10] 人脸五个关键位置点（未启用）

                #   计算分类的损失———置信度
                #   对置信度小于2的正样本和负样本进行掩码，符合条件返回1，不符合条件的返回0.
                category_mask = torch.lt(category_, 2)
                #   对标签中置信度小于2的选择掩码，返回适合条件的结果
                category = torch.masked_select(category_, category_mask)
                #   对预测标签（输出）进行编码，返回符合条件的结构
                output_category = torch.masked_select(output_catagory, category_mask)
                #   对置信度做损失
                cls_loss = self.cls_loss_fn(output_category, category)

                #   计算回归框的损失---偏移量
                #   对置信度大于0的标签进行掩码；负样本不参与计算，负样本没有偏移量；[512,1]
                offset_mask = torch.gt(category_, 0)
                #   例如：[[2]] 选出非负样本的索引：[244]
                offset_index = torch.nonzero(offset_mask)[:, 0]
                #   标签里的偏移量：[244, 4]
                offset = offset_[offset_index]
                #   输出的偏移量：[244, 4]
                output_offset = output_offset[offset_index]
                #   偏移量损失
                offset_loss = self.offset_loss_fn(output_offset, offset)
                #   总损失
                loss = cls_loss + offset_loss

                #   反向传播，优化网络
                self.optimizer.zero_grad()
                #   计算梯度
                loss.backward()
                #   优化网络
                self.optimizer.step()

                #   输出的损失：loss-->GPU-->CPU-->Tensor-->Array
                print("i", i, "loss", loss.cpu().data.numpy(), "cls_loss",
                      cls_loss.cpu().data.numpy(), "offset_loss", offset_loss.cpu().data.numpy())

                #   保存
                if (i + 1) % 1000 == 0:
                    #   保存网络参数
                    torch.save(self.net.state_dict(), self.save_path)
                    print("Hi guys, This epoch is saving success now!")
