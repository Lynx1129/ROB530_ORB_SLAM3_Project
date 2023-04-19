import numpy as np
import cv2
import os

# Load optical flow
optical_flow = np.load("flow/000020_000021_flow.npy") 
depth = np.load("depth_left/000021_left_depth.npy")

# Define camera intrinsics
fx = 320
fy = 320
cx = 320
cy = 240

# Compute normalized optical flow
optical_flow_norm = np.zeros_like(optical_flow)
optical_flow_norm[..., 0] = optical_flow[..., 0] / cx
optical_flow_norm[..., 1] = optical_flow[..., 1] / cy

# Compute essential matrix using 8-point algorithm
pts1 = np.array([[(x, y, depth[y, x]*0.001) for x in range(optical_flow.shape[1])] for y in range(optical_flow.shape[0])])
pts1 = np.reshape(pts1, (optical_flow.shape[0]*optical_flow.shape[1], 3))
pts2 = pts1 + np.concatenate((optical_flow_norm, np.zeros_like(optical_flow_norm)), axis=-1)
pts2 = np.reshape(pts2, (optical_flow.shape[0]*optical_flow.shape[1], 3))
F, mask = cv2.findFundamentalMat(pts1[:, :2], pts2[:, :2], cv2.FM_8POINT)

# Compute camera motion from essential matrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
E = K.T @ F @ K
_, R, t, _ = cv2.recoverPose(E, pts1[:, :2], pts2[:, :2], K)

print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)