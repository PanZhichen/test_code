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
#include <pcl/filters/voxel_grid.h>

using namespace std;

tf::TransformBroadcaster *tfBroadcasterPointer = NULL;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_targ (new pcl::PointCloud<pcl::PointXYZI>);
ros::Publisher cloud_pub;
pcl::VoxelGrid<pcl::PointXYZI> voxelgrid;


void livox_cb(const sensor_msgs::PointCloud2::ConstPtr& lvx_msg)
{
//  static Eigen::Matrix4f curr_to_odom = Eigen::Matrix4f::Identity();
//  if(cloud_targ->empty()){
//    pcl::fromROSMsg(*lvx_msg, *cloud_targ);
//    pcl::io::savePCDFileASCII("/home/dji/livox_ws/pcd1.pcd", *cloud_targ);
//    return;
//  }
  string path = "/home/dji/livox_ws/pcd/demo_";
  static int count = 0;
  string save_path = path + to_string(count) + ".pcd";
  pcl::fromROSMsg(*lvx_msg, *cloud_targ);
  for(size_t i=0; i<cloud_targ->points.size(); ++i)
  {
    float tmp = cloud_targ->points[i].x;
    cloud_targ->points[i].x = -cloud_targ->points[i].y;
    cloud_targ->points[i].y = tmp;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  voxelgrid.setInputCloud(cloud_targ);
  voxelgrid.filter(*downsampled);
  *cloud_targ = *downsampled;
  //------------------------------------
  sensor_msgs::PointCloud2 surroundCloud2;
  pcl::toROSMsg(*cloud_targ, surroundCloud2);
  surroundCloud2.header.frame_id = "velodyne";
  surroundCloud2.header.stamp = ros::Time::now();
  cloud_pub.publish(surroundCloud2);
  //------------------------------------
//  pcl::io::savePCDFileASCII(save_path, *cloud_targ);
  count++;
  cloud_targ->clear();
}
int main (int argc, char** argv)
{
  ros::init(argc, argv, "save_bag");
  ros::NodeHandle nh_icp;

  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  ros::Subscriber livox_sub = nh_icp.subscribe("/cloud", 10, livox_cb);
  cloud_pub = nh_icp.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 3);

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }

 return (0);
}
