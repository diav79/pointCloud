The code takes stereo image pair and disparity map as input (hard coded) and generates point cloud. The disparity map is generated using SPS Stereo. SGBM can also be used here.

Prerequisite: OpenCV, Point Cloud Library (http://pointclouds.org/), and (optional) SPS Stereo (https://github.com/vbodlloyd/StereoSegmentation).

After cloning, perform 'cmake .' and 'make'. To run the code with given data, just do './pointCloud'

To use on custom dataset, provide the paths for image pair and disparity map in the code. Moreover, change the camera parameters and R&T values.

Note: Our experiments suggest results obtained using SPS stereo are better than SGBM.
