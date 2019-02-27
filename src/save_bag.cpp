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

using namespace std;

tf::TransformBroadcaster *tfBroadcasterPointer = NULL;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_targ (new pcl::PointCloud<pcl::PointXYZ>);

void livox_cb(const sensor_msgs::PointCloud2::ConstPtr& lvx_msg)
{
  static Eigen::Matrix4f curr_to_odom = Eigen::Matrix4f::Identity();
  if(cloud_targ->empty()){
    pcl::fromROSMsg(*lvx_msg, *cloud_targ);
    pcl::io::savePCDFileASCII("/home/dji/livox_ws/pcd1.pcd", *cloud_targ);
    return;
  }
}
int main (int argc, char** argv)
{
  ros::init(argc, argv, "save_bag");
  ros::NodeHandle nh_icp;

  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  ros::Subscriber livox_sub = nh_icp.subscribe("/cloud", 1, livox_cb);

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }

 return (0);
}
