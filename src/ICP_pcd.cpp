#include <ros/ros.h>
#include <iostream>
#include <string>
#include <cstring>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>
#include <pcl/search/kdtree.h>
#include <pcl/features/fpfh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <ctime>

using namespace std;
typedef pcl::PointXYZ PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

int main (int argc, char** argv)
{
  if(argc != 4){
    cout << endl << "pcd1+pcd2+save_path"<<endl;
    return 1;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_targ (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(argv[1], *cloud_targ);
  pcl::io::loadPCDFile(argv[2], *cloud_src);
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);

  // Downsample
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointT> grid;
  const float leaf = 0.01f;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (cloud_targ);
  grid.filter (*cloud_targ);
  grid.setInputCloud (cloud_src);
  grid.filter (*cloud_src);

  //remove outliers
  pcl::console::print_highlight ("Remove Outliers...\n");
  pcl::RadiusOutlierRemoval<PointT> outrem;
  outrem.setRadiusSearch(0.05);
  outrem.setMinNeighborsInRadius (8);
  outrem.setInputCloud(cloud_targ);
  outrem.filter (*cloud_targ);
  outrem.setInputCloud(cloud_src);
  outrem.filter (*cloud_src);

  tree1->setInputCloud(cloud_targ);
  tree2->setInputCloud(cloud_src);
  icp.setSearchMethodSource(tree1);
  icp.setSearchMethodTarget(tree2);
  icp.setInputSource(cloud_targ);
  icp.setInputTarget(cloud_src);
//  icp.setMaximumIterations(50);
//  icp.setEuclideanFitnessEpsilon(0.05);//前后两次迭代误差的差值
//  icp.setTransformationEpsilon(1e-6); //上次转换与当前转换的差值；
//  icp.setMaxCorrespondenceDistance(0.1); //忽略在此距离之外的点，对配准影响较大

  pcl::console::print_highlight ("Starting ICP...\n");
  clock_t start,end;
  start  = clock();
  pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
  icp.align(*Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
            icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;
  Eigen::Matrix4f m_4f = icp.getFinalTransformation();

//  pcl::PointCloud<pcl::PointXYZI>::Ptr save_p(new pcl::PointCloud<pcl::PointXYZI>);
//  pcl::PointXYZI tmp_p;
//  for(size_t i=0; i<cloud_targ->points.size(); i++){
//    tmp_p.x = cloud_targ->points[i].x;
//    tmp_p.y = cloud_targ->points[i].y;
//    tmp_p.z = cloud_targ->points[i].z;
//    tmp_p.intensity = 1;
//    save_p->points.push_back(tmp_p);
//
//    Eigen::Vector4f new_p = m_4f * Eigen::Vector4f(cloud_targ->points[i].x, cloud_targ->points[i].y, cloud_targ->points[i].z, 1);
//    tmp_p.x = new_p(0);
//    tmp_p.y = new_p(1);
//    tmp_p.z = new_p(2);
//    tmp_p.intensity = 55;
//    save_p->points.push_back(tmp_p);
//  }
//  for(size_t i=0; i<cloud_src->points.size(); i++){
//    tmp_p.x = cloud_src->points[i].x;
//    tmp_p.y = cloud_src->points[i].y;
//    tmp_p.z = cloud_src->points[i].z;
//    tmp_p.intensity = 10;
//    save_p->points.push_back(tmp_p);
//
////    Eigen::Vector4f new_p = m_4f * Eigen::Vector4f(cloud_src->points[i].x, cloud_src->points[i].y, cloud_src->points[i].z, 1);
////    tmp_p.x = new_p(0);
////    tmp_p.y = new_p(1);
////    tmp_p.z = new_p(2);
////    tmp_p.intensity = 55;
////    save_p->points.push_back(tmp_p);
//  }
////  for(size_t i=0; i<Final.points.size(); i++){
////    tmp_p.x = Final.points[i].x;
////    tmp_p.y = Final.points[i].y;
////    tmp_p.z = Final.points[i].z;
////    tmp_p.intensity = 55;
////    save_p->points.push_back(tmp_p);
////  }
//  save_p->width = 1;
//  save_p->height = save_p->points.size();
//
//  string path = "/home/dji/livox_ws/pcd/ICP_";
//  static int count = 0;
//  string save_path = path + to_string(count) + ".pcd";
//  pcl::io::savePCDFileASCII(argv[3], *save_p);

  if (icp.hasConverged())
  {
    end = clock();
    cout <<"calculate time is: "<< float (end-start)/CLOCKS_PER_SEC<<endl;
    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
//    visu.addPointCloud (cloud_targ, ColorHandlerT (cloud_targ, 0.0, 255.0, 0.0), "cloud_targ");
    visu.addPointCloud (cloud_src, ColorHandlerT (cloud_src, 0.0, 0.0, 255.0), "cloud_src");
    visu.addPointCloud (Final, ColorHandlerT (Final, 255.0, 0.0, 0.0), "Final");
    visu.spin ();
  }
  else
  {
    pcl::console::print_error ("Failed!!!\n");
    return (1);
  }

 return (0);
}
