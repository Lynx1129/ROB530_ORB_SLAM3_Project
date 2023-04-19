import numpy as np
import cv2
import os

from azure.storage.blob import ContainerClient
import io
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Set the camera intrinsic matrix
K = np.array([[616.437378, 0.0, 324.957407], [0.0, 616.437378, 238.688687], [0.0, 0.0, 1.0]])

# Load the optical flow
flow = np.load("flow/000004_000005_flow.npy")

# Intrinsic matrix of the TartanAir dataset
K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])

# Estimate essential matrix from optical flow
F, mask = cv2.findFundamentalMat(flow[:,:,0], flow[:,:,1], cv2.RANSAC, 0.1)
E = K.T @ F @ K

# Compute the rotation and translation from the essential matrix
U, S, Vt = np.linalg.svd(E)
W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
R = U @ W @ Vt
t = U[:,2]
