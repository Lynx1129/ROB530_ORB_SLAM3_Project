import time
new_traj = []
with open("pose_left.txt", "r") as f:
    data = f.readlines()
    for line in data:
        line = line.strip('\n').split()
        line = [float(l) for l in line]
        w = line.pop(-1)
        line.insert(3, w)
        new_traj.append(line)

with open("new_left.txt", 'w') as f:
    for line in new_traj:
        line = [str(l) for l in line]
        pose = " ".join(line)
        curr = str(time.time())
        f.write(curr + " " + pose + "\n")
        