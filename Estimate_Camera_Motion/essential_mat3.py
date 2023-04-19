import cv2
import numpy as np

# this is bull shit

# Load the normalized optical flow and depth maps
flow = np.load("flow/000020_000021_flow.npy") 
depth = np.load("depth_left/000021_left_depth.npy")

# Compute the normalized coordinates of the optical flow vectors
h, w = depth.shape
x, y = np.meshgrid(range(w), range(h))
fx, fy = 320,320  # focal length in pixels, assuming TartanAir's dataset
cx, cy = 320,240 # image center
xn = (x - cx) / fx
yn = (y - cy) / fy
du = flow[..., 0] / fx
dv = flow[..., 1] / fy

# Normalize the optical flow vectors by the corresponding depths
dz = depth[y, x]
du_norm = du / dz
dv_norm = dv / dz

# Stack the normalized coordinates and optical flow vectors
pts1 = np.stack([xn.ravel(), yn.ravel()], axis=1)
pts2 = np.stack([xn.ravel() + du_norm.ravel(), yn.ravel() + dv_norm.ravel()], axis=1)

# Estimate the fundamental matrix using the 8-point algorithm
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # intrinsic matrix
F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# Convert the fundamental matrix to the essential matrix
E = np.dot(K.T, np.dot(F, K))

# Extract the rotation and translation from the essential matrix
U, S, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
if np.linalg.det(Vt) < 0:
    Vt *= -1
R = np.dot(np.dot(U, W), Vt)
t = U[:, 1]

print("Rotation:\n", R)
print("Translation:\n", t)