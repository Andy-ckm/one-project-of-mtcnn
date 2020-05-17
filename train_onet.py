#   训练O网络

import mtcnn_03.nets as nets
import mtcnn_03.train as train

if __name__ == '__main__':
    net = nets.ONet()
    #   网络，参数保存，训练图片的路径
    trainer = train.Trainer(net, "./param_0/onet.pt", r"C:\celeba_2\48")
    #   调用训练方法
    trainer.train()
