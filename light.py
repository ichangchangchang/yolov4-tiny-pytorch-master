from PIL import Image
import numpy as np
from yolo import YOLO


def MinRgb(c):
    return min(c[0], c[1], c[2])


def SumRgb(c):
    return c[0] + c[1] + c[2]


def Invert(img):
    img = 255 - img
    return img


def GetA(R, G, B, k=100):
    # k默认是原文获取排序后前100个像素点
    rlist = []
    height, width = R.shape[0], R.shape[1]
    for hi in range(height):
        for wi in range(width):
            rlist.append([R[hi][wi], G[hi][wi], B[hi][wi]])
    rlist.sort(key=MinRgb)
    rlist.reverse()
    rlist = rlist[:k]
    rlist.sort(key=SumRgb)
    rlist.reverse()
    return rlist[0][0], rlist[0][1], rlist[0][2]


def CalT(R, G, B, r_A, g_A, b_A, size=1, w=0.76):
    # 计算A值时使用size×size窗口，以图像边缘点为窗口中心时需要进行填充
    # 图像填充时上下左右各填充1行/列255
    ts = (size - 1) // 2
    height, width = R.shape[0], R.shape[1]
    R_f = np.pad(R, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / r_A
    G_f = np.pad(G, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / g_A
    B_f = np.pad(B, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / b_A

    shape = (height, width, size, size)
    strides = R_f.itemsize * np.array([width + ts * 2, 1, width + ts * 2, 1])

    blocks_R = np.lib.stride_tricks.as_strided(R_f, shape=shape, strides=strides)
    blocks_G = np.lib.stride_tricks.as_strided(G_f, shape=shape, strides=strides)
    blocks_B = np.lib.stride_tricks.as_strided(B_f, shape=shape, strides=strides)

    t = np.zeros((height, width))
    for hi in range(height):
        for wi in range(width):
            t[hi, wi] = 1 - w * min(np.min(blocks_R[hi, wi]), np.min(blocks_G[hi, wi]), np.min(blocks_B[hi, wi]))
            if t[hi, wi] < 0.5:
                t[hi, wi] = 2 * t[hi, wi] * t[hi, wi]
    return t


def DeHaze(img):
    # 获取图像宽度、高度
    # width, height = img.size
    # 获取图像的RGB数组
    img = np.asarray(img, dtype=np.int32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    #进行反转
    R, G, B = Invert(R), Invert(G), Invert(B)
    # 计算A值
    r_A, g_A, b_A = GetA(R, G, B)
    t = CalT(R, G, B, r_A, g_A, b_A)
    #得到真实图(也就是去雾之后的反转图)
    J_R = (R - r_A) / t + r_A
    J_G = (G - g_A) / t + g_A
    J_B = (B - b_A) / t + b_A
    # 进行低光照图还原，光照增强
    J_R, J_G, J_B = Invert(J_R), Invert(J_G), Invert(J_B)
    r = Image.fromarray(J_R).convert('L')
    g = Image.fromarray(J_G).convert('L')
    b = Image.fromarray(J_B).convert('L')
    image = Image.merge("RGB", (r, g, b))
    image.save("dark_result.jpg")
    image.show()


if __name__ == '__main__':
    yolo = YOLO()
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            DeHaze(image)
            img = Image.open("dark_result.jpg")
            r_image = yolo.detect_image(img)
            r_image.show()
            r_image.save('7.jpg')
