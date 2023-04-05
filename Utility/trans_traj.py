import numpy as np
import transformation as tf
import time
def cam2ned(traj):
    '''
    transfer a camera traj to ned frame traj
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)

def ned2cam(traj):
    '''
    transfer a ned traj to camera frame traj
    '''
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))
    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)
with open("sample_001/mav0/pose_left.txt", 'r') as f:
    data = f.readlines()

with open("f_dataset-tartan_stereo.txt", "r") as d:
    pre_pose = d.readlines()

# traj = []
# for line in pre_pose:
#     line = line.strip("\n")
#     line = line.split(" ")
#     line = [float(l) for l in line][1:]
#     traj.append(line)
# new_traj = cam2ned(traj)
# with open("ned_traj_tartan.txt", "w") as w:
#     for line in new_traj:
#         line = [str(l) for l in line]
#         curr = str(time.time())
#         w.write(curr + " " + " ".join(line) + '\n')
    
traj = []
for line in pre_pose:
    line = line.strip("\n")
    line = line.split(" ")
    line = [float(l) for l in line][1:]
    traj.append(line)
with open("_traj_tartan.txt", "w") as w:
    for line in traj:
        line = [str(l) for l in line]
        curr = str(time.time())
        w.write(curr + " " + " ".join(line) + '\n')

    