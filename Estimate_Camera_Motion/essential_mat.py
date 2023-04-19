import numpy as np
import cv2

# Define calibration matrix K (replace with your own values)
K = np.array([[320, 0, 320],
              [0, 320, 240],
              [0, 0, 1]])

# Load images and flow data
flow = np.load("flow/000000_000001_flow.npy")  # assumes flow data is stored in .flo format

# Normalize flow data
du = flow[:, :, 0]/640
dv = flow[:, :, 1]/480
pts1 = np.vstack((du.flatten(), dv.flatten()))
pts2 = pts1 + np.array([[1], [0]])  # create second set of points shifted by (1, 0)

# Compute essential matrix using eight-point algorithm
pts1_norm = np.linalg.inv(K) @ np.vstack((pts1, np.ones((1, pts1.shape[1]))))
pts2_norm = np.linalg.inv(K) @ np.vstack((pts2, np.ones((1, pts2.shape[1]))))
#E, _ = cv2.findEssentialMat(pts1_norm[:2].T, pts2_norm[:2].T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

print("pts1_norm's shape is ", pts1_norm.shape)

F, mask = cv2.findFundamentalMat(pts1_norm[:2].T, pts2_norm[:2].T, cv2.RANSAC, 0.1)
E = K.T @ F @ K

# Ensure essential matrix satisfies constraint on singular values
U, S, Vt = np.linalg.svd(E)
S = np.array([1, 1, 0])
E = U @ np.diag(S) @ Vt

# Recover relative camera pose from essential matrix
retval, R, t, _ = cv2.recoverPose(E, pts1_norm[:2].T, pts2_norm[:2].T, K)

print("Essential matrix:")
print(E)
print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)