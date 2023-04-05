import time
from os import listdir, rename
from os.path import join


right_path = "./sample_001/mav0/cam1/data"
left_path = "./sample_001/mav0/cam0/data"
left_files = [f for f in listdir(left_path)]
right_files = [f for f in listdir(right_path)]
left_files.sort()
right_files.sort()
with open("tartanStamp.txt", "w") as f:
    for i in range(len(left_files)):
        curr = str(time.time())
        f.write(curr+'\n')
        print(left_files[i])
        rename(join(right_path, right_files[i]), join(right_path, curr+'.png'))
        rename(join(left_path, left_files[i]), join(left_path, curr+'.png'))
