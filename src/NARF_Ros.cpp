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
pcl::search::KdTree<pcl::PointWithRange>::Ptr tree(new pcl::search::KdTree<pcl::PointWithRange>);
Eigen::Matrix4f curr_to_odom = Eigen::Matrix4f::Identity();
PointCloudT::Ptr CloudPub (new PointCloudT);
ros::Publisher cloud_pub;
tf::TransformBroadcaster *tfBroadcasterPointer = NULL;

class EstimateMotion
{
public:
    EstimateMotion(float supportsize, float leafin);
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
};
EstimateMotion::EstimateMotion(float supportsize, float leafin) {
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
void EstimateMotion::livox_cb(const sensor_msgs::PointCloud2::ConstPtr& msg){
  if(keypoints_target->points.empty()){
    PointCloudT::Ptr cloud_(new PointCloudT);
    pcl::RangeImage rangeImage;
    pcl::PointCloud<int> keypoint_indices;
    PointCloudNT::Ptr Normal_(new PointCloudNT);

    pcl::fromROSMsg(*msg, *cloud_);
    for(size_t i=0; i<cloud_->points.size(); ++i){
      float tmp = cloud_->points[i].x;
      cloud_->points[i].x = cloud_->points[i].y;
      cloud_->points[i].y = -tmp;
    }
    voxelfilter(cloud_);
    radiusfilter(cloud_);
    genRangeImage(rangeImage, cloud_);
    detectNARF(&rangeImage, keypoint_indices);
    computeNormal(rangeImage, Normal_);
    pcl::PointIndicesPtr kkeypoints_t (new pcl::PointIndices);
    for(size_t i=0; i<keypoint_indices.points.size (); i++){
      kkeypoints_t->indices.push_back(keypoint_indices.points[i]);
    }
    computeFeature(tree, rangeImage, Normal_, kkeypoints_t, features_target);
    pcl::PointCloud<pcl::PointXYZ>& keypoints_t = *keypoints_target;
    keypoints_t.points.resize (keypoint_indices.points.size ());
    for (size_t i=0; i<keypoint_indices.points.size (); ++i)
      keypoints_t.points[i].getVector3fMap () = rangeImage.points[keypoint_indices.points[i]].getVector3fMap ();
    return;
  }

  PointCloudT::Ptr cloud_s(new PointCloudT);
  pcl::RangeImage rangeImage_s;
  pcl::PointCloud<int> keypoint_indices_s;
  PointCloudNT::Ptr Normal_s(new PointCloudNT);
  PointCloudT::Ptr cloud_aligned(new PointCloudT);
  size_t inliers = 0;
  Eigen::Matrix4f transformation;

  pcl::fromROSMsg(*msg, *cloud_s);
  for(size_t i=0; i<cloud_s->points.size(); ++i){
    float tmp = cloud_s->points[i].x;
    cloud_s->points[i].x = cloud_s->points[i].y;
    cloud_s->points[i].y = -tmp;
  }
  voxelfilter(cloud_s);
  radiusfilter(cloud_s);
  genRangeImage(rangeImage_s, cloud_s);
  detectNARF(&rangeImage_s, keypoint_indices_s);
  computeNormal(rangeImage_s, Normal_s);
  pcl::PointIndicesPtr kkeypoints_s (new pcl::PointIndices);
  for(size_t i=0; i<keypoint_indices_s.points.size (); i++){
    kkeypoints_s->indices.push_back(keypoint_indices_s.points[i]);
  }
  computeFeature(tree, rangeImage_s, Normal_s, kkeypoints_s, features_source);
  pcl::PointCloud<pcl::PointXYZ>& keypoints_s = *keypoints_source;
  keypoints_s.points.resize (keypoint_indices_s.points.size ());
  for (size_t i=0; i<keypoint_indices_s.points.size (); ++i)
    keypoints_s.points[i].getVector3fMap () = rangeImage_s.points[keypoint_indices_s.points[i]].getVector3fMap ();
  inliers = doAlignment(keypoints_source,keypoints_target,features_source,features_target,cloud_aligned,transformation);
  if(inliers>0){
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", inliers, keypoints_source->size ());

    curr_to_odom = curr_to_odom * transformation;
    Eigen::Quaterniond q_eigen(curr_to_odom.block<3,3>(0,0).cast<double>());
    Eigen::Vector3d v(curr_to_odom.block<3,1>(0,3).cast<double>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*cloud_s, *transformed_cloud, curr_to_odom);
    for(size_t i=0; i<transformed_cloud->size(); ++i){
      CloudPub->points.push_back(transformed_cloud->points[i]);
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

    PointCloudT::Ptr keypoints_tmp = keypoints_target;
    FeatureCloudT::Ptr features_tmp = features_target;

    keypoints_target = keypoints_source;
    keypoints_source = keypoints_tmp;

    features_target = features_source;
    features_source = features_tmp;
  }
  keypoints_source->clear();
  features_source->clear();
}

int main (int argc, char** argv) {
  ros::init(argc, argv, "NARF_Ros");
  ros::NodeHandle nh;

  EstimateMotion estimatmotion(0.2f, 0.01f);
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

//  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new PointCloudT);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_s(new PointCloudT);
////  PointCloudNT::Ptr pointcloudNT(new PointCloudNT);
////  PointCloudNT::Ptr pointcloudNT_s(new PointCloudNT);
//
//  // Get input cloud
//  if (argc != 3)
//  {
//    pcl::console::print_error ("Syntax is: %s load1.pcd load2.pcd\n", argv[0]);
//    return (1);
//  }
//  // Load object and scene
//  pcl::console::print_highlight ("Loading point clouds...\n");
//  if (pcl::io::loadPCDFile<PointT> (argv[1], *pointCloud) < 0
//      || pcl::io::loadPCDFile<PointT> (argv[2], *pointCloud_s) < 0)
//  {
//    pcl::console::print_error ("Error loading object/scene file!\n");
//    return (1);
//  }
//
//  // Downsample
//  estimatmotion.voxelfilter(pointCloud);
//  estimatmotion.voxelfilter(pointCloud_s);
//  //remove outliers
//  estimatmotion.radiusfilter(pointCloud);
//  estimatmotion.radiusfilter(pointCloud_s);
//  //rang image
//  pcl::RangeImage rangeImageSource, rangeImageTarget;
//  estimatmotion.genRangeImage(rangeImageSource, pointCloud);
//  std::cout << rangeImageSource << "\n\n";
//  estimatmotion.genRangeImage(rangeImageTarget, pointCloud_s);
//  std::cout << rangeImageTarget << "\n\n";
//  //extract NARF
//  pcl::PointCloud<int> keypoint_indices_s, keypoint_indices_t;
//  estimatmotion.detectNARF(&rangeImageSource, keypoint_indices_s);
//  std::cout << "Found "<<keypoint_indices_s.points.size ()<<" key points.\n";
//  estimatmotion.detectNARF(&rangeImageTarget, keypoint_indices_t);
//  std::cout << "Found "<<keypoint_indices_t.points.size ()<<" key points.\n";
//  //compute normal
//  PointCloudNT::Ptr NT_source (new PointCloudNT);
//  PointCloudNT::Ptr NT_target(new PointCloudNT);
//  estimatmotion.computeNormal(rangeImageSource, NT_source);
//  estimatmotion.computeNormal(rangeImageTarget, NT_target);
//  // Estimate features
//  pcl::search::KdTree<pcl::PointWithRange>::Ptr tree(new pcl::search::KdTree<pcl::PointWithRange>);
//  FeatureCloudT::Ptr cloud_features_source (new FeatureCloudT);
//  FeatureCloudT::Ptr cloud_features_target (new FeatureCloudT);
//  pcl::PointIndicesPtr kkeypoints_s (new pcl::PointIndices);
//  for(size_t i=0; i<keypoint_indices_s.points.size (); i++){
//    kkeypoints_s->indices.push_back(keypoint_indices_s.points[i]);
//  }
//  pcl::PointIndicesPtr kkeypoints_t (new pcl::PointIndices);
//  for(size_t i=0; i<keypoint_indices_t.points.size (); i++){
//    kkeypoints_t->indices.push_back(keypoint_indices_t.points[i]);
//  }
//  estimatmotion.computeFeature(tree, rangeImageSource, NT_source, kkeypoints_s, cloud_features_source);
//  std::cout<<"cloud features:="<<cloud_features_source->points.size()<<std::endl;
//  estimatmotion.computeFeature(tree, rangeImageTarget, NT_target, kkeypoints_t, cloud_features_target);
//  std::cout<<"cloud features:="<<cloud_features_target->points.size()<<std::endl;
//  // Perform alignment
//  PointCloudT::Ptr cloud_aligned (new PointCloudT);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_s (new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZ>& keypoints_s = *keypoints_ptr_s;
//  keypoints_s.points.resize (keypoint_indices_s.points.size ());
//  for (size_t i=0; i<keypoint_indices_s.points.size (); ++i)
//    keypoints_s.points[i].getVector3fMap () = rangeImageSource.points[keypoint_indices_s.points[i]].getVector3fMap ();
//
//  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_t (new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZ>& keypoints_t = *keypoints_ptr_t;
//  keypoints_t.points.resize (keypoint_indices_t.points.size ());
//  for (size_t i=0; i<keypoint_indices_t.points.size (); ++i)
//    keypoints_t.points[i].getVector3fMap () = rangeImageTarget.points[keypoint_indices_t.points[i]].getVector3fMap ();
//  Eigen::Matrix4f transformation;
//  size_t inliers = estimatmotion.doAlignment(keypoints_ptr_s,keypoints_ptr_t,cloud_features_source,
//                                             cloud_features_target,cloud_aligned,transformation);
//  if(inliers>0){
//    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
//    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
//    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
//    pcl::console::print_info ("\n");
//    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
//    pcl::console::print_info ("\n");
//    pcl::console::print_info ("Inliers: %i/%i\n", inliers, keypoints_ptr_s->size ());
//
//    // Executing the transformation
//    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
//    pcl::transformPointCloud (*pointCloud, *transformed_cloud, transformation);
//    pcl::visualization::PCLVisualizer visu("Alignment");
//    visu.addPointCloud (pointCloud_s, ColorHandlerT (pointCloud_s, 0.0, 255.0, 0.0), "target");
//    visu.addPointCloud (transformed_cloud, ColorHandlerT (transformed_cloud, 0.0, 0.0, 255.0), "source");
//    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target");
//    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
//    visu.spin ();
//  }
  return(0);
}


