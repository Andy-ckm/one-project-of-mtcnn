#   训练p网络

import nets as nets
import train as train

if __name__ == '__main__':
    net = nets.PNet()
    #   网络，参数保存，训练图片的路径
    trainer = train.Trainer(net, "./param_0/pnet.pt", r"C:\celeba_2\12")
    #   调用训练方法
    trainer.train()
