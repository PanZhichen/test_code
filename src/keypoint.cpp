#include <Eigen/Core>
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

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<pcl::PointWithRange,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

bool live_update = false;
float support_size = 0.2f; //0.2

void
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

int main (int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new PointCloudT);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_s(new PointCloudT);
  PointCloudNT::Ptr pointcloudNT(new PointCloudNT);
  PointCloudNT::Ptr pointcloudNT_s(new PointCloudNT);
  PointCloudNT::Ptr NT_1 (new PointCloudNT);
  PointCloudNT::Ptr NT_s(new PointCloudNT);
  pcl::search::KdTree<pcl::PointWithRange>::Ptr tree(new pcl::search::KdTree<pcl::PointWithRange>);
  FeatureCloudT::Ptr cloud_features (new FeatureCloudT);
  FeatureCloudT::Ptr cloud_features_s (new FeatureCloudT);
  PointCloudT::Ptr cloud_aligned (new PointCloudT);

  // Get input cloud
  if (argc != 3)
  {
    pcl::console::print_error ("Syntax is: %s load1.pcd load2.pcd\n", argv[0]);
    return (1);
  }
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointT> (argv[1], *pointCloud) < 0
          || pcl::io::loadPCDFile<PointT> (argv[2], *pointCloud_s) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }
  clock_t start,end;
  start  = clock();
  // Downsample
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointT> grid;
  const float leaf = /*0.005f;*/0.01f;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (pointCloud);
  grid.filter (*pointCloud);
  grid.setInputCloud (pointCloud_s);
  grid.filter (*pointCloud_s);

  //remove outliers
  pcl::console::print_highlight ("Remove Outliers...\n");
  pcl::RadiusOutlierRemoval<PointT> outrem;
  outrem.setRadiusSearch(0.04);
  outrem.setMinNeighborsInRadius (3);
  outrem.setInputCloud(pointCloud);
  outrem.filter (*pointCloud);
  outrem.setInputCloud(pointCloud_s);
  outrem.filter (*pointCloud_s);
//  pcl::io::savePCDFileASCII("/home/dji/livox_ws/pcd/22222.pcd", *pointCloud);
//  pcl::io::savePCDFileASCII("/home/dji/livox_ws/pcd/11111.pcd", *pointCloud_s);

  // create a range image
  float angularResolution = (float) (  0.2f * (M_PI/180.0f));  //   1.0 degree in radians
  float maxAngleWidth     = (float) (50.0f * (M_PI/180.0f));  // 360.0 degree in radians
  float maxAngleHeight    = (float) (50.0f * (M_PI/180.0f));  // 180.0 degree in radians
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
  float noiseLevel=0.00;
  float minRange = 0.0f;
  int borderSize = 1;
  pcl::console::print_highlight ("Range Image...\n");
  pcl::RangeImage rangeImage, rangeImage_s;
  rangeImage.createFromPointCloud(*pointCloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
  std::cout << rangeImage << "\n\n";
  rangeImage_s.createFromPointCloud(*pointCloud_s, angularResolution, maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
  std::cout << rangeImage_s << "\n\n";

  // --------------------------------
  // -----Extract NARF keypoints-----
  // --------------------------------
  pcl::RangeImageBorderExtractor range_image_border_extractor;
  pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
  narf_keypoint_detector.setRangeImage (&rangeImage);
  narf_keypoint_detector.getParameters ().support_size = support_size;
  narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true; //true
  narf_keypoint_detector.getParameters ().min_interest_value = 0.5; //0.6
  narf_keypoint_detector.getParameters ().optimal_range_image_patch_size = 20;
  //narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;
//  std::cout<< "min_interest_value:="
//    << narf_keypoint_detector.getParameters ().min_interest_value<<std::endl;
//  std::cout<< "optimal_range_image_patch_size:="
//    << narf_keypoint_detector.getParameters ().optimal_range_image_patch_size<<std::endl;
//  std::cout<< "distance_for_additional_points:="
//    << narf_keypoint_detector.getParameters ().distance_for_additional_points<<std::endl;
  pcl::PointCloud<int> keypoint_indices;
  narf_keypoint_detector.compute (keypoint_indices);
  std::cout << "Found "<<keypoint_indices.points.size ()<<" key points.\n";

  narf_keypoint_detector.setRangeImage (&rangeImage_s);
  pcl::PointCloud<int> keypoint_indices_s;
  narf_keypoint_detector.compute (keypoint_indices_s);
  std::cout << "Found "<<keypoint_indices_s.points.size ()<<" key points.\n";

  // Estimate normals for scene
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<pcl::PointWithRange,PointNT> nest;
  nest.setRadiusSearch (0.03);
  nest.setInputCloud (rangeImage.makeShared());
  nest.compute (*NT_1);
  nest.setInputCloud (rangeImage_s.makeShared());
  nest.compute (*NT_s);

  // Estimate features
  pcl::PointIndicesPtr kkeypoints (new pcl::PointIndices);
  for(size_t i=0; i<keypoint_indices.points.size (); i++){
    kkeypoints->indices.push_back(keypoint_indices.points[i]);
  }
  pcl::PointIndicesPtr kkeypoints_s (new pcl::PointIndices);
  for(size_t i=0; i<keypoint_indices_s.points.size (); i++){
    kkeypoints_s->indices.push_back(keypoint_indices_s.points[i]);
  }
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch /*(0.025);*/(0.08);
  fest.setSearchMethod(tree);
  fest.setInputCloud (rangeImage.makeShared());
  fest.setInputNormals (NT_1);
  fest.setIndices(kkeypoints);
  fest.compute (*cloud_features);
  std::cout<<"cloud features:="<<cloud_features->points.size()<<std::endl;
  fest.setInputCloud (rangeImage_s.makeShared());
  fest.setInputNormals (NT_s);
  fest.setIndices(kkeypoints_s);
  fest.compute (*cloud_features_s);
  std::cout<<"cloud features_s:="<<cloud_features_s->points.size()<<std::endl;

  // Perform alignment
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;
  keypoints.points.resize (keypoint_indices.points.size ());
  for (size_t i=0; i<keypoint_indices.points.size (); ++i)
    keypoints.points[i].getVector3fMap () = rangeImage.points[keypoint_indices.points[i]].getVector3fMap ();

  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_s (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& keypoints_s = *keypoints_ptr_s;
  keypoints_s.points.resize (keypoint_indices_s.points.size ());
  for (size_t i=0; i<keypoint_indices_s.points.size (); ++i)
    keypoints_s.points[i].getVector3fMap () = rangeImage_s.points[keypoint_indices_s.points[i]].getVector3fMap ();

  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<PointT,PointT,FeatureT> align;
  align.setInputSource (keypoints_ptr);
  align.setSourceFeatures (cloud_features);
  align.setInputTarget (keypoints_ptr_s);
  align.setTargetFeatures (cloud_features_s);
  align.setMaximumIterations (20000); // Number of RANSAC iterations
  align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (5); // Number of nearest features to use
  align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (2.5f); // Inlier threshold
  align.setInlierFraction (0.4f); // Required inlier fraction for accepting a pose hypothesis
  align.setEuclideanFitnessEpsilon(0.02);//前后两次迭代误差的差值
  align.setTransformationEpsilon(1e-9); //上次转换与当前转换的差值；
  {
    pcl::ScopeTime t("Alignment");
    align.align (*cloud_aligned);
  }

  if (align.hasConverged ())
  {
    end = clock();
    cout <<"calculate time is: "<< float (end-start)/CLOCKS_PER_SEC<<endl;
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), keypoints_ptr->size ());

    //imply ICP....
//    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
//    tree1->setInputCloud(pointCloud);
//    tree2->setInputCloud(pointCloud_s);
//    icp.setSearchMethodSource(tree1);
//    icp.setSearchMethodTarget(tree2);
//    icp.setInputSource(pointCloud);
//    icp.setInputTarget(pointCloud_s);
//    icp.setMaxCorrespondenceDistance(0.5);
//    icp.setTransformationEpsilon(1e-9);
//    icp.setEuclideanFitnessEpsilon(0.01);
//    icp.setMaximumIterations (100);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
//    icp.align(*Final, transformation);
    // Show alignment

    // Executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*pointCloud, *transformed_cloud, transformation);
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (pointCloud_s, ColorHandlerT (pointCloud_s, 0.0, 255.0, 0.0), "target");
    visu.addPointCloud (transformed_cloud, ColorHandlerT (transformed_cloud, 0.0, 0.0, 255.0), "source");
    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target");
    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
//    visu.addPointCloud (cloud_aligned, ColorHandlerT (cloud_aligned, 0.0, 0.0, 255.0), "object_aligned");
//    visu.addPointCloud (Final, ColorHandlerT (Final, 255.0, 0.0, 0.0), "Final");
//    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Final");
    visu.spin ();
  }

  // -------------------------------------
  // -----Show keypoints in 3D viewer-----
  // -------------------------------------
//  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
//  viewer.setBackgroundColor (1, 1, 1);
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler
//          (rangeImage.makeShared(), 0, 0, 0);
//  viewer.addPointCloud (rangeImage.makeShared(), range_image_color_handler, "range image");
//  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "range image");
//  viewer.initCameraParameters ();
//  setViewerPose(viewer, rangeImage.getTransformationToWorldSystem ());
//
////  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
////  pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;
////  keypoints.points.resize (keypoint_indices.points.size ());
////  for (size_t i=0; i<keypoint_indices.points.size (); ++i)
////    keypoints.points[i].getVector3fMap () = rangeImage.points[keypoint_indices.points[i]].getVector3fMap ();
//
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypoints_ptr, 0, 255, 0);
//  viewer.addPointCloud<pcl::PointXYZ> (keypoints_ptr, keypoints_color_handler, "keypoints");
//  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
//  //--------------------------------------
//  pcl::visualization::PCLVisualizer viewer_s ("3D Viewer_s");
//  viewer_s.setBackgroundColor (1, 1, 1);
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler_s
//          (rangeImage_s.makeShared(), 0, 0, 0);
//  viewer_s.addPointCloud (rangeImage_s.makeShared(), range_image_color_handler_s, "range image_s");
//  viewer_s.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "range image_s");
//  viewer_s.initCameraParameters ();
//  setViewerPose(viewer_s, rangeImage_s.getTransformationToWorldSystem ());
//
////  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_s (new pcl::PointCloud<pcl::PointXYZ>);
////  pcl::PointCloud<pcl::PointXYZ>& keypoints_s = *keypoints_ptr_s;
////  keypoints_s.points.resize (keypoint_indices_s.points.size ());
////  for (size_t i=0; i<keypoint_indices_s.points.size (); ++i)
////    keypoints_s.points[i].getVector3fMap () = rangeImage_s.points[keypoint_indices_s.points[i]].getVector3fMap ();
//
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler_s (keypoints_ptr_s, 0, 255, 0);
//  viewer_s.addPointCloud<pcl::PointXYZ> (keypoints_ptr_s, keypoints_color_handler_s, "keypoints_s");
//  viewer_s.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_s");
//
//  //--------------------
//  // -----Main loop-----
//  //--------------------
//  while (!viewer.wasStopped () && !viewer_s.wasStopped ())
//  {
////    range_image_widget.spinOnce ();
//    viewer.spinOnce ();
//    viewer_s.spinOnce();
//    pcl_sleep (0.01);
//  }
}

