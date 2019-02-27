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
    return;
  }

  pcl::fromROSMsg(*lvx_msg, *cloud_src);
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
  tree1->setInputCloud(cloud_targ);
  tree2->setInputCloud(cloud_src);
  icp.setSearchMethodSource(tree1);
  icp.setSearchMethodTarget(tree2);

  icp.setInputSource(cloud_targ);
  icp.setInputTarget(cloud_src);
  icp.setMaximumIterations(50);
  icp.setEuclideanFitnessEpsilon(0.5);//前后两次迭代误差的差值
  icp.setTransformationEpsilon(1e-6); //上次转换与当前转换的差值；
  icp.setMaxCorrespondenceDistance(0.1); //忽略在此距离之外的点，对配准影响较大

  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
            icp.getFitnessScore() << std::endl;

  pcl::PointCloud<pcl::PointXYZI>::Ptr save_p(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI tmp_p;
  for(size_t i=0; i<cloud_targ->points.size(); i++){
    tmp_p.x = cloud_targ->points[i].x;
    tmp_p.y = cloud_targ->points[i].y;
    tmp_p.z = cloud_targ->points[i].z;
    tmp_p.intensity = 1;
    save_p->points.push_back(tmp_p);
  }
  for(size_t i=0; i<cloud_src->points.size(); i++){
    tmp_p.x = cloud_src->points[i].x;
    tmp_p.y = cloud_src->points[i].y;
    tmp_p.z = cloud_src->points[i].z;
    tmp_p.intensity = 10;
    save_p->points.push_back(tmp_p);
  }
  for(size_t i=0; i<Final.points.size(); i++){
    tmp_p.x = Final.points[i].x;
    tmp_p.y = Final.points[i].y;
    tmp_p.z = Final.points[i].z;
    tmp_p.intensity = 55;
    save_p->points.push_back(tmp_p);
  }
  save_p->width = 1;
  save_p->height = save_p->points.size();

  string path = "/home/dji/livox_ws/pcd/ICP_";
  static int count = 0;
  string save_path = path + to_string(count) + ".pcd";
  pcl::io::savePCDFileASCII(save_path, *save_p);
  ++count;
  Eigen::Matrix4f m_4f = icp.getFinalTransformation();
//  std::cout << icp.getFinalTransformation() << std::endl;

  curr_to_odom = m_4f * curr_to_odom;
  Eigen::Quaterniond q_eigen(curr_to_odom.block<3,3>(0,0).cast<double>());
  Eigen::Vector3d v(curr_to_odom.block<3,1>(0,3).cast<double>());

  tf::StampedTransform voTrans;
  voTrans.frame_id_ = "odom";
  voTrans.child_frame_id_ = "sensor_frame";
  voTrans.stamp_ = ros::Time::now();
  voTrans.setRotation(tf::Quaternion(q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w()));
  voTrans.setOrigin(tf::Vector3(v(0), v(1), v(2)));
  tfBroadcasterPointer->sendTransform(voTrans);

  pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_ptr;
  tmp_ptr = cloud_src;
  cloud_src = cloud_targ;
  cloud_targ = tmp_ptr;
  cloud_src->clear();
}
int main (int argc, char** argv)
{
  ros::init(argc, argv, "icp_test");
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
