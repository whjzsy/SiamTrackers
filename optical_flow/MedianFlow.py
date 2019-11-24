#！/home/ubuntu/anaconda3/bin python
# -*- coding:utf-8 -*-
"""
    【--------------------------Median_Flow----------------------】
    title{Forward-Backward Error: Automatic Detection of Tracking Failures}
"""
import cv2
import numpy as np
lk_params = dict(winSize  = (11, 11),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

def medianflow(pre_image, cur_image, center_pos, size):
    n_samples = 100
    fb_max_dist = 1
    ds_factor = 0.95
    min_n_points = 10

    # get gray image
    pre_image = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
    cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)

    # get bbox
    bbox = [center_pos[0] - size[0] / 2,
                center_pos[1] - size[1] / 2,
                size[0],
                size[1]]

    # sample points inside the bounding box
    p0 = np.empty((n_samples, 2))
    p0[:, 0] = np.random.randint(bbox[0], bbox[0] + bbox[2] + 1, n_samples)
    p0[:, 1] = np.random.randint(bbox[1], bbox[1] + bbox[3] + 1, n_samples)

    p0 = p0.astype(np.float32)

    # forward-backward tracking
    p1, st, err = cv2.calcOpticalFlowPyrLK(pre_image, cur_image, p0, None, **lk_params)
    indx = np.where(st == 1)[0]
    p0 = p0[indx, :]
    p1 = p1[indx, :]
    p0r, st, err = cv2.calcOpticalFlowPyrLK(cur_image, pre_image, p1, None, **lk_params)
    if err is None:

        return center_pos[0], center_pos[1]

    # check forward-backward error and min number of points
    fb_dist = np.abs(p0 - p0r).max(axis=1)
    good = fb_dist < fb_max_dist

    # keep half of the points
    err = err[good].flatten()
    if len(err) < min_n_points:

        return center_pos[0], center_pos[1]

    indx = np.argsort(err)
    half_indx = indx[:len(indx) // 2]
    p0 = (p0[good])[half_indx]
    p1 = (p1[good])[half_indx]

    # estimate displacement
    dx = np.median(p1[:, 0] - p0[:, 0])
    dy = np.median(p1[:, 1] - p0[:, 1])

    cx = center_pos[0] + dx
    cy = center_pos[1] + dy

    return cx, cy
