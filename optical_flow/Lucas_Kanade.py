#！/home/ubuntu/anaconda3/bin python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7,
                      useHarrisDetector=False,
                      k = 0.04)

def Lk_optical_flow(imgA, imgB, center_pos, s_x):

    # 获取imgA帧的角点
    imgA_gray = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    imgB_gray = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 获取需要获取角点的mask区域， 注意选着mask区域的越界问题
    im_sz = imgA_gray.shape
    # left = int(center_pos[0] - size[0]/2 + 0.5)
    # top = int(center_pos[1] - size[1]/2 + 0.5)
    # right = int(left + size[0] - 1)
    # bottom = int(top + size[1] - 1)
    #
    c = (s_x + 1) / 2
    context_xmin = np.floor(center_pos[0] - c + 0.5)
    context_xmax = context_xmin + s_x - 1
    context_ymin = np.floor(center_pos[1] - c + 0.5)
    context_ymax = context_ymin + s_x - 1

    left = int(max(0, context_xmin))
    top = int(max(0, context_ymin))
    right = int(min(im_sz[1], context_xmax))
    bottom = int(min(im_sz[0], context_ymax))

    mask = np.zeros_like(imgA_gray)
    mask[top:(bottom + 1), left:(right + 1)] = 255   #选择的区域检测角点
    # mask = np.zeros_like(imgA_gray)
    # mask[:] = 255
    # tracks = []
    p0 = cv2.goodFeaturesToTrack(imgA_gray, mask=mask, **feature_params)  # 像素级别角点检测
    # if p is not None:
    #     for x, y in np.float32(p).reshape(-1, 2):
    #         tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
    #
    # p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    # p0 = cv2.cornerSubPix(imgA_gray, p0, winSize=(7, 7), zeroZone=(-1, -1),
    #                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if not (p0 is None):
        p1, st, err = cv2.calcOpticalFlowPyrLK(imgA_gray, imgB_gray, p0, None, **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置

        p0r, st, err = cv2.calcOpticalFlowPyrLK(imgB_gray, imgA_gray, p1, None, **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置

        d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
        good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点

        x = 0
        y = 0
        count = 0
        for (x0, y0), (x1, y1), good_flag in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            x += x1 - x0
            y += y1 - y0
            count += 1
    else:
        count = 0

    #需要注意count为0的情况
    if count == 0:
        x_avg = 0
        y_avg = 0
    else:
        x_avg = x // count
        y_avg = y // count

    # 注意图片中的x为纵轴， y为横轴
    cx = center_pos[0] + x_avg
    cy = center_pos[1] + y_avg

    # 防止中心坐标超出界限

    return cx, cy

if __name__ == '__main__':


    # print(-3 // 2)
    video_src = '/home/ubuntu/Desktop/Object_Track/SiamTrackers/demo/bag.avi'
    cam = cv2.VideoCapture(video_src)
    first_frame = True
    video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    while True:
        ret, frame = cam.read()
        if ret == True:

            if first_frame == True:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                first_frame = False
                center_pos = np.array([init_rect[0] + (init_rect[2] - 1) / 2,
                                       init_rect[1] + (init_rect[3] - 1) / 2])
                size = np.array([init_rect[2], init_rect[3]])
                preframe = frame
            else:
                pass


