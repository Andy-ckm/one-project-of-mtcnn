#   训练R网络

import nets as nets
import train as train

if __name__ == '__main__':
    net = nets.RNet()
    #   网络，参数保存，训练图片的路径
    trainer = train.Trainer(net, "./param_0/rnet.pt", r"C:\celeba_2\24")
    #   调用训练方法
    trainer.train()
