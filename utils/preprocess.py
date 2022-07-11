'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-11 16:49:57
Email: haimingzhang@link.cuhk.edu.cn
Description: The preprocessing of the input image.
'''

import numpy as np
import cv2
from scipy.io import loadmat
import pickle


class DataProcessor(object):
    def __init__(self, lm3d_fp="../BFM/similarity_Lm3D_all.mat") -> None:
        self.lm3d = load_lm3d(lm3d_fp)
    
    def __call__(self, img, ldmk):
        img, lm_affine, mat = Preprocess(img, ldmk, self.lm3d)

        mat_inv = np.array([[1 / mat[0, 0], 0, -mat[0, 2] / mat[0, 0]],
                            [0, 1 / mat[1, 1], -mat[1, 2] / mat[1, 1]]])
        return img, lm_affine, mat, mat_inv


def load_lm3d(lm3d_fp):
    Lm3D = loadmat(lm3d_fp)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def POS(xp, x):
    npts = xp.shape[0]
    if npts == 68:
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5
    if npts == 29:
        lm_idx = np.array([20, 8, 10, 9, 11, 22, 23])
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5

    A = np.zeros([2 * npts, 8])
    x = np.concatenate((x, np.ones((npts, 1))), axis=1)
    A[0:2 * npts - 1:2, 0:4] = x

    A[1:2 * npts:2, 4:] = x

    b = np.reshape(xp, [-1, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def process_img(img, t, s, target_size=256):
    h0, w0 = img.shape[:2]
    scale = 116. / s
    dx = -(t[0, 0] * scale - target_size / 2)
    dy = -((h0 - t[1, 0]) * scale - target_size / 2)
    mat = np.array([[scale, 0, dx],
                    [0, scale, dy]])

    corners = np.array([[0, 0, 1], [w0 - 1, h0 - 1, 1]])
    new_corners = (corners @ mat.T).astype('int32')
    pad_left = max(new_corners[0, 0], 0)
    pad_top = max(new_corners[0, 1], 0)
    pad_right = min(new_corners[1, 0], target_size - 1)
    pad_bottom = min(new_corners[1, 1], target_size - 1)
    mask = np.zeros((target_size, target_size, 3))
    mask[:pad_top, :, :] = 1
    mask[pad_bottom:, :, :] = 1
    mask[:, :pad_left, :] = 1
    mask[:, pad_right:, :] = 1
    img_affine = cv2.warpAffine(img, mat, (target_size, target_size), borderMode=cv2.BORDER_REFLECT_101)
    img_affine = img_affine.astype('float32') * (1 - mask) + cv2.blur(img_affine, (10, 10)) * mask
    img_affine = img_affine.astype('uint8')
    return img_affine, mat


def Preprocess(img, lm, lm3D):
    h0, w0 = img.shape[:2]
    lm_ = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)
    t, s = POS(lm_, lm3D)
    img_new, mat = process_img(img, t, s)
    lm_affine = np.concatenate((lm[:, :2], np.ones((lm.shape[0], 1))), axis=1)
    lm_affine = lm_affine @ mat.T
    return img_new, lm_affine, mat


def load_landmarks(lm_path):
    if lm_path.endswith(".pkl"):
        with open(lm_path, 'rb') as f:
            landmark = pickle.load(f)
    elif lm_path.endswith(".txt"):
        landmark = np.loadtxt(lm_path)
    elif lm_path.endswith(".npy"):
        landmark = np.load(lm_path)
    else:
        raise ValueError("Unknown file type")
    return landmark



if __name__ == "__main__":
    img_path = "/home/zhanghm/Research/V100/TalkingFaceFormer/data/HDTF_face3dmmformer/train/WDA_BarackObama_000/face_image/000000.jpg"
    ldmk_path = "/home/zhanghm/Research/V100/TalkingFaceFormer/data/HDTF_face3dmmformer/train/WDA_BarackObama_000/face_image/000000.txt"

    img = cv2.imread(img_path)
    ldmk = load_landmarks(ldmk_path)

    data_processor = DataProcessor()

    img, lm_affine, mat, _ = data_processor(img, ldmk)
    print(img.shape)
    # cv2.imwrite("temp.jpg", img)
    print(mat.shape, mat)
