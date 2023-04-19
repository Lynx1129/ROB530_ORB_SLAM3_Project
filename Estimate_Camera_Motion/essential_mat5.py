import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load optical flow
flow = np.load("flow/000020_000021_flow.npy") 
depth = np.load("depth_left/000021_left_depth.npy")

# Normalize optical flow by pixel size
pixel_size = 0.0039  # TartanAir dataset pixel size
flow_norm = flow / pixel_size

# Compute essential matrix
K = np.array([[320, 0.0, 320], [0.0, 320, 240], [0.0, 0.0, 1.0]])  # TartanAir intrinsic matrix
E, _ = cv2.findEssentialMat(flow_norm, depth, K, cv2.RANSAC, 0.999, 1.0, None)

# Decompose essential matrix to obtain rotation and translation matrices
_, R, t, _ = cv2.recoverPose(E, flow_norm, depth, K)
print(R)
print(t)