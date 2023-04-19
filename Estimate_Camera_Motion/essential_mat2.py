import numpy as np
import cv2

# Load optical flow and depth files
flow = np.load("flow/000002_000003_flow.npy")  # assumes flow data is stored in .flo format
depth = np.load("depth_left/000000_left_depth.npy")

# Define calibration matrix K (replace with your own values)
K = np.array([[320, 0, 320],
              [0, 320, 240],
              [0, 0, 1]])

# Extract du and dv components from optical flow
du = flow[..., 0].astype(float)
dv = flow[..., 1].astype(float)

# Normalize du and dv components by pixel size and focal length
pixel_size = 0.0028 # pixel size in meters (change this to the actual value for your dataset)
focal_length = 320 # focal length in pixels (change this to the actual value for your dataset)
du_norm = du / (pixel_size * focal_length)
dv_norm = dv / (pixel_size * focal_length)

print(dv_norm.shape)

# Convert optical flow to correspondences
corres = []
for i in range(flow.shape[0]):
    for j in range(flow.shape[1]):
        corres.append([(j+flow[i,j,0])/du_norm[i,j], (i+flow[i,j,1])/dv_norm[i,j]])

# Normalize correspondences
corres = np.array(corres)
corres_mean = np.mean(corres, axis=0)
corres_std = np.std(corres, axis=0)
corres_norm = (corres - corres_mean) / corres_std

# Compute essential matrix using 8-point algorithm
F, mask = cv2.findFundamentalMat(corres_norm[:8], corres_norm[8:], cv2.FM_8POINT)
E = K.T @ F @ K

# # Estimate essential matrix from normalized optical flow
# E, mask = cv2.findEssentialMat(np.reshape(du_norm, (-1, 1)), np.reshape(dv_norm, (-1, 1)), K, cv2.RANSAC, 0.999, 1.0)

# # Extract rotation and translation matrices from essential matrix
# R1, R2, t = cv2.decomposeEssentialMat(E)

# # Recover camera motion from rotation and translation matrices and depth data
# h, w = depth.shape
# x, y = np.meshgrid(np.arange(w), np.arange(h))
# x = x.astype(float)
# y = y.astype(float)
# X = x * depth / focal_length
# Y = y * depth / focal_length
# Z = depth
# points1 = np.stack((X, Y, Z), axis=-1)
# points2 = np.stack((X + du_norm, Y + dv_norm, Z), axis=-1)
# points1 = np.reshape(points1, (-1, 3))
# points2 = np.reshape(points2, (-1, 3))
# points1 = points1[mask.ravel() == 1]
# points2 = points2[mask.ravel() == 1]
# _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

# # Print camera motion
# print("Rotation matrix:\n", R)
# print("Translation vector:\n", t)
