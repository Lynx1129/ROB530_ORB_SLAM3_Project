import numpy as np
from scipy.spatial.transform import Rotation

# Load the optical flow and mask from the npy files
flow_npy = np.load("path/to/flow.npy")
mask_npy = np.load("path/to/mask.npy")

# Define the initial rotation quaternion and translation vector
q = np.array([1, 0, 0, 0], dtype=np.float32)
t = np.array([0, 0, 0], dtype=np.float32)

# Define the scale factor for the rotation angle
scale = 1.0

# Loop over each pair of optical flow and mask
for flow, mask in zip(flow_npy, mask_npy):
    # Calculate the mean flow vector in the x and y directions
    fx = np.mean(flow[..., 0][mask > 0])
    fy = np.mean(flow[..., 1][mask > 0])

    # Calculate the rotation angle from the flow vector
    angle = np.arctan2(fy, fx) * scale

    # Convert the rotation angle to a quaternion
    rot = Rotation.from_rotvec([0, 0, angle])
    q_new = rot.as_quat()

    # Integrate the rotation quaternion over time
    q = q * q_new

    # Integrate the flow vector over time
    t += np.array([fx, fy, 0], dtype=np.float32)

    # Convert the cumulative rotation quaternion and translation vector to a transformation matrix
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(q).as_matrix()
    mat[:3, 3] = t

    # Extract the translation and rotation from the transformation matrix
    translation = mat[:3, 3]
    rotation = Rotation.from_matrix(mat[:3, :3]).as_quat()

    # Print the odometry values
    print("Translation: ", translation)
    print("Rotation: ", rotation)
