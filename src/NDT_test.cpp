#include <Eigen/Core>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>
#include <ctime>
#include <string>
#include <cstring>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>
#include <pcl/registration/ndt.h>

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<pcl::PointWithRange,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

PointCloudT::Ptr keypoints_source (new PointCloudT);
PointCloudT::Ptr keypoints_target (new PointCloudT);
FeatureCloudT::Ptr features_source (new FeatureCloudT);
FeatureCloudT::Ptr features_target (new FeatureCloudT);
pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::search::KdTree<pcl::PointWithRange>::Ptr tree(new pcl::search::KdTree<pcl::PointWithRange>);
Eigen::Matrix4f curr_to_odom = Eigen::Matrix4f::Identity();
PointCloudT::Ptr CloudPub (new PointCloudT);
ros::Publisher cloud_pub;
tf::TransformBroadcaster *tfBroadcasterPointer = NULL;

class EstimateMotion
{
public:
    EstimateMotion(float supportsize, float leafin,
            pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr& ndt_omp_ptr);
    ~EstimateMotion(){};
    void voxelfilter(PointCloudT::Ptr& cloud);
    void radiusfilter(const PointCloudT::Ptr& cloud);
    void genRangeImage(pcl::RangeImage& rangeimage, PointCloudT::Ptr& cloud);
    void detectNARF(pcl::RangeImage* rangeimage, pcl::PointCloud<int>& keypoint_indices);
    void computeNormal(pcl::RangeImage& rangeimage, PointCloudNT::Ptr& normalcloud);
    void computeFeature(pcl::search::KdTree<pcl::PointWithRange>::Ptr& tree,
                        pcl::RangeImage &rangeimage, PointCloudNT::Ptr& normalcloud,
                        pcl::PointIndicesPtr& kkeypoints, FeatureCloudT::Ptr& cloud_features);
    size_t doAlignment(PointCloudT::Ptr& source, PointCloudT::Ptr& target, FeatureCloudT::Ptr& feature_source,
                       FeatureCloudT::Ptr& feature_target, PointCloudT::Ptr& cloud_aligned, Eigen::Matrix4f& transformation);
    void livox_cb(const sensor_msgs::PointCloud2::ConstPtr& msg);
    bool NDT_Align(const pcl::Registration<PointT, PointT>::Ptr& registration,
                                              const PointCloudT::Ptr& target_cloud,
                                              const PointCloudT::Ptr& source_cloud,
                                              Eigen::Matrix4f& transformation);
private:
    float support_size_, leaf_;
    pcl::VoxelGrid<PointT> grid_;
    pcl::RadiusOutlierRemoval<PointT> outrem_;
    // create a range image
    float angularResolution_ = (float) (  0.2f * (M_PI/180.0f));  //   1.0 degree in radians
    float maxAngleWidth_     = (float) (50.0f * (M_PI/180.0f));  // 360.0 degree in radians
    float maxAngleHeight_    = (float) (50.0f * (M_PI/180.0f));  // 180.0 degree in radians
    Eigen::Affine3f sensorPose_ = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame_ = pcl::RangeImage::LASER_FRAME;
    float noiseLevel_=0.00;
    float minRange_ = 0.0f;
    int borderSize_ = 1;
    pcl::RangeImageBorderExtractor range_image_border_extractor_;
    pcl::NarfKeypoint narf_keypoint_detector_;
    pcl::NormalEstimationOMP<pcl::PointWithRange,PointNT> nest_;
    FeatureEstimationT fest_;
    pcl::SampleConsensusPrerejective<PointT,PointT,FeatureT> align_;
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp_;
};
EstimateMotion::EstimateMotion(float supportsize, float leafin,
        pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr& ndt_omp_ptr) {
  ndt_omp_ = ndt_omp_ptr;
  support_size_ = supportsize;
  leaf_ = leafin;
  grid_.setLeafSize (leaf_, leaf_, leaf_);
  outrem_.setRadiusSearch(0.04);
  outrem_.setMinNeighborsInRadius (3);
  narf_keypoint_detector_.setRangeImageBorderExtractor(&range_image_border_extractor_);
  narf_keypoint_detector_.getParameters ().support_size = support_size_;
  narf_keypoint_detector_.getParameters ().add_points_on_straight_edges = true; //true
  narf_keypoint_detector_.getParameters ().min_interest_value = 0.5; //0.6
  narf_keypoint_detector_.getParameters ().optimal_range_image_patch_size = 20;
  //narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;
  nest_.setRadiusSearch (0.03);
  fest_.setRadiusSearch /*(0.025);*/(0.08);

  align_.setMaximumIterations (20000); // Number of RANSAC iterations
  align_.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  align_.setCorrespondenceRandomness (5); // Number of nearest features to use
  align_.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
  align_.setMaxCorrespondenceDistance (2.5f); // Inlier threshold
  align_.setInlierFraction (0.4f); // Required inlier fraction for accepting a pose hypothesis
  align_.setEuclideanFitnessEpsilon(0.02);//前后两次迭代误差的差值
  align_.setTransformationEpsilon(1e-9); //上次转换与当前转换的差值；
}
void EstimateMotion::voxelfilter(PointCloudT::Ptr &cloud) {
  PointCloudT::Ptr downsample(new PointCloudT);
  grid_.setInputCloud (cloud);
  grid_.filter (*downsample);
  *cloud = *downsample;
}
void EstimateMotion::radiusfilter(const PointCloudT::Ptr &cloud) {
  outrem_.setInputCloud(cloud);
  outrem_.filter (*cloud);
}
void EstimateMotion::genRangeImage(pcl::RangeImage &rangeimage, PointCloudT::Ptr& cloud) {
  rangeimage.createFromPointCloud(*cloud, angularResolution_, maxAngleWidth_, maxAngleHeight_,
                                  sensorPose_, coordinate_frame_, noiseLevel_, minRange_, borderSize_);
}
void EstimateMotion::detectNARF(pcl::RangeImage* rangeimage, pcl::PointCloud<int> &keypoint_indices) {
  narf_keypoint_detector_.setRangeImage(rangeimage);
  narf_keypoint_detector_.compute (keypoint_indices);
}
void EstimateMotion::computeNormal(pcl::RangeImage& rangeimage, PointCloudNT::Ptr& normalcloud) {
  nest_.setInputCloud (rangeimage.makeShared());
  nest_.compute (*normalcloud);
}
void EstimateMotion::computeFeature(pcl::search::KdTree<pcl::PointWithRange>::Ptr& tree,
                                    pcl::RangeImage &rangeimage, PointCloudNT::Ptr& normalcloud,
                                    pcl::PointIndicesPtr& kkeypoints, FeatureCloudT::Ptr& cloud_features) {
  fest_.setSearchMethod(tree);
  fest_.setInputCloud (rangeimage.makeShared());
  fest_.setInputNormals (normalcloud);
  fest_.setIndices(kkeypoints);
  fest_.compute (*cloud_features);
}
size_t EstimateMotion::doAlignment(PointCloudT::Ptr &source, PointCloudT::Ptr &target, FeatureCloudT::Ptr &feature_source,
                                   FeatureCloudT::Ptr &feature_target, PointCloudT::Ptr &cloud_aligned,
                                   Eigen::Matrix4f& transformation) {
  align_.setInputSource (source);
  align_.setSourceFeatures (feature_source);
  align_.setInputTarget (target);
  align_.setTargetFeatures (feature_target);
  align_.align (*cloud_aligned); //这里可以提供一个Matrix4形式的guess
  if (align_.hasConverged ()){
    std::cout<<"WIN!!!!!"<<std::endl;
    transformation = align_.getFinalTransformation ();

    return align_.getInliers ().size ();
  }else{
    return 0;
  }
}
bool EstimateMotion::NDT_Align(const pcl::Registration<PointT, PointT>::Ptr& registration,
                                           const PointCloudT::Ptr &target_cloud, const PointCloudT::Ptr &source_cloud,
                                           Eigen::Matrix4f &transformation) {
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
  auto t1 = ros::WallTime::now();
  registration->align(*aligned);

  auto t2 = ros::WallTime::now();
  std::cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;
  if(registration->hasConverged()){
    transformation = registration->getFinalTransformation();
    return true;
  }else{
    return false;
  }
}
void EstimateMotion::livox_cb(const sensor_msgs::PointCloud2::ConstPtr& msg){
  if(target_cloud->points.empty()){
//    PointCloudT::Ptr cloud_(new PointCloudT);
    pcl::fromROSMsg(*msg, *target_cloud);
    for(size_t i=0; i<target_cloud->points.size(); ++i){
      float tmp = target_cloud->points[i].x;
      target_cloud->points[i].x = target_cloud->points[i].y;
      target_cloud->points[i].y = -tmp;
    }
    voxelfilter(target_cloud);
    return;
  }

//  PointCloudT::Ptr cloud_s(new PointCloudT);
  pcl::RangeImage rangeImage_s;
  pcl::PointCloud<int> keypoint_indices_s;
  PointCloudNT::Ptr Normal_s(new PointCloudNT);
  PointCloudT::Ptr cloud_aligned(new PointCloudT);
  size_t inliers = 0;
  Eigen::Matrix4f transformation;

  pcl::fromROSMsg(*msg, *source_cloud);
  for(size_t i=0; i<source_cloud->points.size(); ++i){
    float tmp = source_cloud->points[i].x;
    source_cloud->points[i].x = source_cloud->points[i].y;
    source_cloud->points[i].y = -tmp;
  }
  voxelfilter(source_cloud);
  if(NDT_Align(ndt_omp_, target_cloud, source_cloud, transformation)){
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", inliers, keypoints_source->size ());


    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_tmp (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*source_cloud, *transformed_cloud_tmp, transformation);
    std::string path = "/home/dji/livox_ws/pcd/NDT_";
    static int count = 0;
    std::string save_path = path + std::to_string(count) + ".pcd";
    pcl::PointCloud<pcl::PointXYZI> save_p;
    pcl::PointXYZI tmp_xyzi;
    for (auto &point : transformed_cloud_tmp->points) {
      tmp_xyzi.x = point.x;
      tmp_xyzi.y = point.y;
      tmp_xyzi.z = point.z;
      tmp_xyzi.intensity = 10;
      save_p.points.push_back(tmp_xyzi);
    }
    for (auto &point : target_cloud->points) {
      tmp_xyzi.x = point.x;
      tmp_xyzi.y = point.y;
      tmp_xyzi.z = point.z;
      tmp_xyzi.intensity = 60;
      save_p.points.push_back(tmp_xyzi);
    }
    save_p.width = 1;
    save_p.height = save_p.points.size();
    pcl::io::savePCDFileASCII(save_path, save_p);
    count++;


    curr_to_odom = curr_to_odom * transformation;
    Eigen::Quaterniond q_eigen(curr_to_odom.block<3,3>(0,0).cast<double>());
    Eigen::Vector3d v(curr_to_odom.block<3,1>(0,3).cast<double>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*source_cloud, *transformed_cloud, curr_to_odom);
    for(size_t i=0; i<transformed_cloud->size(); ++i){
      CloudPub->points.push_back(transformed_cloud->points[i]);
    }
    if(CloudPub->points.size()>100000){
      PointCloudT::Ptr downsample_t(new PointCloudT);
      grid_.setInputCloud (CloudPub);
      grid_.filter (*downsample_t);
      *CloudPub = *downsample_t;
    }
    sensor_msgs::PointCloud2 surroundCloud2;
    pcl::toROSMsg(*CloudPub, surroundCloud2);
    surroundCloud2.header.frame_id = "odom";
    surroundCloud2.header.stamp = ros::Time::now();
    cloud_pub.publish(surroundCloud2);

    tf::StampedTransform voTrans;
    voTrans.frame_id_ = "odom";
    voTrans.child_frame_id_ = "sensor_frame";
    voTrans.stamp_ = ros::Time::now();
    voTrans.setRotation(tf::Quaternion(q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w()));
    voTrans.setOrigin(tf::Vector3(v(0), v(1), v(2)));
    tfBroadcasterPointer->sendTransform(voTrans);

    PointCloudT::Ptr keypoints_tmp = target_cloud;
    target_cloud = source_cloud;
    source_cloud = keypoints_tmp;
  }
  source_cloud->clear();
}

int main (int argc, char** argv) {
  ros::init(argc, argv, "NDT_test");
  ros::NodeHandle nh;

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp->setResolution(1.0);
  ndt_omp->setNumThreads(8);
  ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  EstimateMotion estimatmotion(0.2f, 0.1f, ndt_omp);
  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  ros::Subscriber livox_sub = nh.subscribe("/cloud", 30, &EstimateMotion::livox_cb, &estimatmotion);
  cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloudMap", 3);

  ros::Rate loop_rate(10);
  while(ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }
  return(0);
}


