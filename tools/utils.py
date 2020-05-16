import numpy as np
import matplotlib.pyplot as plt


#   iou:一种除以最小框的面积，一种除以交集的面积
def iou(box, boxes, isMin = False):
    #   计算面积：[x1, y1, x2, y2]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 找交集
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 找交集的面积
    inter = w * h
    if isMin:
        #   最小面积的IOU：在O网络中使用
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        #   交集/并集的IOU：P 和 R 中使用
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr


# 非极大值抑制（mns）
def nms(boxes, thresh=0.3, isMin = False):
    #   当框的长度为0时，防止报错,返回空
    if boxes.shape[0] == 0:
        return np.array([])

    #   当框的长度不为0时
    #   根据置信度排序：[x1,y1.x2,y2]
    #   根据置信度，由大到小排序
    _boxes = boxes[(-boxes[:, 4]).argsort()]

    #   创建空列表。存放保留剩余的框
    r_boxes = []
    #   使用排完序的第一个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）
    while _boxes.shape[0] > 1:
        #   shape[0]等价于shape(0),代表0轴上框的个数（维度）
        #   取出第一个框
        a_box = _boxes[0]
        #   取出剩余的框
        b_boxes = _boxes[1:]

        #   将第一个框加入列表
        #   每循环一次，往列表添加一个框
        r_boxes.append(a_box)

        #   比较IOU，将符合阈值条件的框保留下来
        #   返回保留框的索引
        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        #   取出符合条件的建议框
        _boxes = b_boxes[index]

    #   剩余最后一个，应该保留
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


#   找到中心点，及最大边长。沿着最大边长的两边扩充
def convert_to_square(bbox):
    #   将长方向框，补齐成正方形框
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    #   框高
    h = bbox[:, 3] - bbox[:, 1]
    #   框宽
    w = bbox[:, 2] - bbox[: 0]
    #   返回最大变长
    max_size = np.maximum(h, w)
    #   对[x1, y1, x2, y2]进行赋值
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_size * 0.5
    square_bbox[:, 1] = bbox[:, 1] + w * 0.5 - max_size * 0.5
    square_bbox[:, 2] = bbox[:, 2] + max_size
    square_bbox[:, 3] = bbox[:, 3] + max_size
    return square_bbox

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def show_rect(bs):
    fig, ax = plt.subplot()
    for i in bs:
        #   长方形
        rect = plt.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
    show_rect(bs)
    print(nms(bs))
    show_rect(nms(bs))