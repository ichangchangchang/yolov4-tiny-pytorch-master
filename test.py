# --------------------------------------------#
#   查看网络参数和结构
# --------------------------------------------#
import torch
from torchsummary import summary

from nets.Dw_yolo4_tiny import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(3, 20).to(device)
    summary(model, input_size=(3, 416, 416))

    # darknet
    # Total
    # params: 5, 918, 006
    # Trainable
    # params: 5, 918, 006

    # mobilenet1:
    # Total
    # params: 2, 842, 230
    # Trainable
    # params: 2, 842, 230

    # Total
    # params: 4, 075, 926

    # 深度可分级卷积_darknet
    # Total
    # params: 4, 878, 903
    # Trainable
    # params: 4, 878, 903
    # Non - trainable
    # params: 0

    # 深度可分离卷积最终版_
    # Total params: 2,787,383
    # Trainable params: 2,787,383
    # Non-trainable params: 0