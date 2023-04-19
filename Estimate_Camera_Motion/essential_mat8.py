import cv2
import numpy as np

# read the rgb image, depth map, and optical flow files
flow = np.load("flow/000000_000001_flow.npy") 
depth1 = np.load("depth_left/000000_left_depth.npy")
depth2 = np.load("depth_left/000001_left_depth.npy")

# normalize the optical flow values
flow = flow.astype(np.float32)
flow[:,:,0] /= 640
flow[:,:,1] /= 480

# extract corresponding 3D points from the depth map
fx = 320
fy = 320
cx = 320
cy = 240

x, y = np.meshgrid(np.arange(depth1.shape[1]), np.arange(depth1.shape[0]))
points3D = np.stack([x, y, np.zeros((depth1.shape[0],depth1.shape[1]))], axis=-1)    
#points3D = np.stack([x, y, depth1], axis=-1)   
points3D = points3D.reshape(-1, 3)
points3D[:, 0] = (points3D[:, 0] - cx) / fx
points3D[:, 1] = (points3D[:, 1] - cy) / fy

points3D2 = np.stack([x, y, np.zeros((depth1.shape[0],depth1.shape[1]))], axis=-1)    
#points3D2 = np.stack([x, y, depth2], axis=-1)   
points3D2 = points3D.reshape(-1, 3)
points3D2[:, 0] = (points3D[:, 0] - cx) / fx
points3D2[:, 1] = (points3D[:, 1] - cy) / fy

#print(points3D[:,0].shape)


# compute the fundamental matrix using the 8-point algorithm
pts1 = points3D[:, :3]  # 2D points in the first image
pts2 = points3D2[:,:3] + np.hstack([flow.reshape(-1, 2),np.zeros(307200).reshape(-1,1)])  # corresponding 2D points in the second image
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# print(pts1.shape)
# print(flow.reshape(-1,2).shape)


# decompose the essential matrix from the fundamental matrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
E = np.matmul(np.matmul(K.T, F), K)

# # Ensure essential matrix satisfies constraint on singular values
# U, S, Vt = np.linalg.svd(E)
# S = np.array([1, 1, 0])
# E = U @ np.diag(S) @ Vt

U, S, Vt = np.linalg.svd(E)
W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
Winv = np.array([[0,1,0],[-1,0,0],[0,0,1]])

R = np.matmul(U,np.matmul(W,Vt))
t = U[:,1]

print(R)
print(t)

#obtain the 3D camera motion
T = -np.matmul(R, t)


# # simple test
# T = np.hstack([R,t])
# T = np.vstack([T,np.array([0,0,0,1])])
# pose0 = np.array([7.008041858673095703e+00, -3.013244628906250000e+01, -3.011430501937866211e+00, 1])
# pose1 = np.matmul(T,pose0)
# print("pose1 is : ", pose1)

#print(T)