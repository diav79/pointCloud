#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <cstdio>
#include <iostream>
#include <fstream>
#include "pcl/common/common_headers.h"
#include "pcl/io/io.h"
#include "pcl/visualization//pcl_visualizer.h"
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include "pcl/point_cloud.h"
#include "pcl/visualization/cloud_viewer.h"

using namespace cv;
using namespace std;
using namespace pcl;


ofstream out("points.txt");

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


int main()
{
    Mat img1, img2;

    Mat actualDisp;

    img1 = imread("I0.png");
    img2 = imread("I2.png");

    actualDisp = imread("actual.png", CV_LOAD_IMAGE_GRAYSCALE);

    Mat g1,g2, disp, disp8;

    // cvtColor(img1, g1, CV_BGR2GRAY);
    // cvtColor(img2, g2, CV_BGR2GRAY);
    g1 = img1;
    g2 = img2;

    int sadSize = 9;
    StereoSGBM sbm;
    sbm.SADWindowSize = sadSize;
    sbm.numberOfDisparities = 112;//144; 128
    sbm.preFilterCap = 31; //63
    sbm.minDisparity = -40; //-39; 0
    sbm.uniquenessRatio = 7.0;
    sbm.speckleWindowSize = 100;
    sbm.speckleRange = 16;
    sbm.disp12MaxDiff = 0;
    sbm.fullDP = true;
    sbm.P1 = 600; // sadSize*sadSize*4;
    sbm.P2 = 2400; // sadSize*sadSize*32;
    sbm(g1, g2, disp);

    // disp8 = disp;

    /// normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    // normalize(actualDisp, actualDisp, 0, 255, CV_MINMAX, CV_8U);

    // Mat dispSGBMscale; 

    disp.convertTo(disp,CV_32F, 1./16); 

    double cm1[3][3] = {{376.9671088772267, 0.000000e+00, 320.5448049510822}, {0.000000e+00, 382.9147550105959, 171.1628286009299}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double cm2[3][3] = {{376.9671088772267, 0.000000e+00, 320.5448049510822}, {0.000000e+00, 382.9147550105959, 171.1628286009299}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double d1[1][5] = {{ -0.01406068420214969, 0.001009405080742544, -0.01228835985255173, 0.0002401701324276044, 0.00}};
    double d2[1][5] = {{ -0.01406068420214969, 0.001009405080742544, -0.01228835985255173, 0.0002401701324276044, 0.00}};

    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);

    // double r[3][3] = {{0.999604, -0.000792602, -0.0281338},{0.000795164, 1.0000, 7.99093e-05 },{0.0281338, -0.000102249, 0.999604}};
    // double t[3][1] = {{- 0.06158932072 + 0.09992132469}, {0.6346606603 - 0.6380563662}, {- 1.191050695 + 1.184780603}};

    double r_ot[3][3] = {{0.999817, -0.00110485, 0.0191098}, {0.00117805, 0.999992, -0.00381951}, {-0.0191054, 0.00384133, 0.99981}};
    double t_ot[3][1] = {{-0.0142182153 + 0.09992132469}, {-0.6261671357 + 0.6380563662}, {1.198082424 - 1.184780603}};

    // Mat R (3,3, CV_64FC1, r);
    // Mat T (3,1, CV_64FC1, t);

    Mat R (3,3, CV_64FC1, r_ot);
    Mat T (3,1, CV_64FC1, t_ot);

    //Mat   R, T;
    Mat R1, R2, T1, T2, Q, P1, P2;
    Mat map1_1, map2_1, map1_2, map2_2;
    Mat dst1, dst2;

    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    
    // cout << Q << "\n";

    // initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_16SC2, map1_1, map2_1);
    // remap(g1, dst1, map1_1, map2_1, INTER_LINEAR, BORDER_CONSTANT, 0);

    // initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_16SC2, map1_2, map2_2);
    // remap(g2, dst2, map1_2, map2_2, INTER_LINEAR, BORDER_CONSTANT, 0);

    // imwrite("dst1.jpg", dst1);
    // imwrite("dst2.jpg", dst2);

    Mat points, points1;
    // reprojectImageTo3D(disp, points, Q, true);
    reprojectImageTo3D(actualDisp, points, Q, true);
    imshow("points", points);
    cvtColor(points, points1, CV_BGR2GRAY);
    imshow("points1", points1);

    // imwrite("disparity.jpg", disp8);
    imwrite("points1.jpg", points1);
    imwrite("points.jpg", points);


    for(int i=0; i<points.rows; ++i)
    {
        Point3f* point = points.ptr<Point3f>(i) ;
        for(int j=0; j<points.cols; ++j)
        {
            out<<i<<" "<<j<<"  x: "<<(*point).x<<" y: "<<(*point).y<<" z: "<<(*point).z<<endl;
            ++point;
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); 

    for (int rows = 0; rows < points.rows; ++rows) { 
        for (int cols = 0; cols < points.cols; ++cols) { 
            cv::Point3f point = points.at<cv::Point3f>(rows, cols); 


            pcl::PointXYZ pcl_point(point.x, point.y, point.z); // normal PointCloud 
            pcl::PointXYZRGB pcl_point_rgb;
            pcl_point_rgb.x = point.x;    // rgb PointCloud 
            pcl_point_rgb.y = point.y; 
            pcl_point_rgb.z = point.z; 
            cv::Vec3b intensity = img1.at<cv::Vec3b>(rows,cols); //BGR 
            uint32_t rgb = (static_cast<uint32_t>(intensity[2]) << 16 | static_cast<uint32_t>(intensity[1]) << 8 | static_cast<uint32_t>(intensity[0])); 
            pcl_point_rgb.rgb = *reinterpret_cast<float*>(&rgb);

            cloud_xyz->push_back(pcl_point); 
            cloud_xyzrgb->push_back(pcl_point_rgb); 
           } 
        } 

     // std::cout << "saving a pointcloud to out.pcd\n";

  //   //Create visualizer
  // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  // viewer = simpleVis( cloud_xyz );

  // //Main loop
  // while ( !viewer->wasStopped())
  // {
  //  viewer->spinOnce(100);
  //  boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  // }

   visualization::CloudViewer viewer("Simple Cloud Viewer");
   viewer.showCloud(cloud_xyzrgb);

    waitKey(0);

    return 0;
}
