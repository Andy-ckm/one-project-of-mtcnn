# one_project_of_mtcnn

# 运行环境准备：
   * win10 + pytorch1.4.0 + Anaconda3(python3.7) + CUDA10.2 + CUDNN7.0

   * 关于配置，可以参考以下两篇博客，两位博主写得很详细：
   * https://blog.csdn.net/weixin_42158966/article/details/88543668
   * https://blog.csdn.net/qq_37296487/article/details/83028394

# 测试结果:

![](photo1.jpg.jpg)
![](photo2.jpg.jpg)
![](photo3.jpg.jpg)
![](photo4.jpg.jpg)
![](photo5.jpg.jpg)


# 如何使用:

  * run > python detect.py

# 使用的数据集:

  * CelebA数据集：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 可以选择官网提供的百度云盘进行下载
  * 下载后在img文件夹解压img_align_celeba_png.7z即可
  * 对应的人脸标签在Anno文件夹下
  * run > sampling.py 处理好训练需要的数据

# 训练:

  * 准备好p网络需要的数据（12 * 12 的图片）
    * run > python train_pnet.py
    * run > python train_rnet.py
    * run > python train_onet.py

