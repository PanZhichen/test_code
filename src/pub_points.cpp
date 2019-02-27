#include <ros/ros.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>

int main (int argc, char** argv)
{
  ros::init(argc, argv, "pub_points");
  ros::NodeHandle n;
  ros::Publisher chatter_pub = n.advertise<sensor_msgs::PointCloud2>("/cloud", 10);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the CloudIn data
  cloud_in->width    = 5;
  cloud_in->height   = 1;
  cloud_in->is_dense = false;
  cloud_in->points.resize (cloud_in->width * cloud_in->height);
  for (size_t i = 0; i < cloud_in->points.size (); ++i)
  {
    cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
  }
  *cloud_out = *cloud_in;
  for (size_t i = 0; i < cloud_in->points.size (); ++i)
    std::cout << "    " <<cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<cloud_in->points[i].z << std::endl;

  ros::Rate loop_rate(5);
  int count = 1;
  while (ros::ok())
  {
    for (size_t i = 0; i < cloud_in->points.size (); ++i){
      cloud_out->points[i].x = cloud_in->points[i].x + 0.01f*count;
      cloud_out->points[i].y = cloud_in->points[i].y + 0.01f*count;
    }

    for (size_t i = 0; i < cloud_out->points.size (); ++i)
      std::cout << "    " <<cloud_out->points[i].x << " " << cloud_out->points[i].y << " " <<cloud_out->points[i].z << std::endl;
    std::cout<< std::endl<< std::endl;
    sensor_msgs::PointCloud2 pcl_2;
    pcl::toROSMsg(*cloud_out, pcl_2);
    pcl_2.header.frame_id = "sensor_frame";
    pcl_2.header.stamp = ros::Time::now();
    chatter_pub.publish(pcl_2);
    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  }
  return (0);
}