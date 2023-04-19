###########################################################################################
What is it for:
it's  for extracting camera motions from dense optical flow provided by Tartanair dataset

How to use:
download Tartanair dataset(for example: abandonedfactory_sample_P001) and extract to this directory, then move the 'flow' folder to this directory. 
You can also refer to pose.txt for comparison.

What are each py files for:
basically all the files are using essential matrix method to extract camera motion from optical flow, the final version we used for GTsam to is essential_mat6.py.

The other files have some modifications on how to normalize the points or whether add depth info or not


