#include <Eigen/Core>
#include <cmath>
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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <vector>
#include <list>
#include <fstream>

using namespace std;
// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int findMaxNormal(vector<int>& NV)
{
  int maxnormal = 0, maxNum = 0;
  for(int i=0; i<NV.size(); i++){
    if(NV[i]>maxNum){
      maxnormal = i;
      maxNum = NV[i];
    }
  }
  return maxnormal;
}

// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // Point clouds
  vector<PointCloudNT> cloudVec;
  cloudVec.resize(30);
  // Get input object and scene
  if (argc != 2)
  {
    pcl::console::print_error ("Syntax is: %s object.pcd\n", argv[0]);
    return (1);
  }
  // Load object and scene
//  pcl::console::print_highlight ("Loading point clouds...\n");
//  if (pcl::io::loadPCDFile<PointNT> (argv[1], *object) < 0){
//    pcl::console::print_error ("Error loading object/scene file!\n");
//    return (1);
//  }

  ifstream in;
  in.open(argv[1], ios::in);
  if (! in.is_open())
  { cout << "Error opening file"; exit (1); }
  string buff;
  while(getline(in, buff)){
    PointNT point;
    vector<float> nums;
    char* s_input = (char*)buff.c_str();
    const char* split = ",";
    char *p= strtok(s_input, split);
    float a;
    while(p!=NULL){
      a = atof(p);
      nums.push_back(a);
      p = strtok(NULL, split);
    }
    point.x = nums[0];
    point.y = nums[1];
    point.z = nums[2];
    cloudVec[nums[3]].points.push_back(point);
  }
  in.close();

  PointCloudNT::Ptr object = cloudVec[0].makeShared();
  cout<<"before"<<object->points.size()<<endl;
  PointCloudNT::Ptr downsampled(new PointCloudNT);
  pcl::VoxelGrid<PointNT> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxelgrid.setInputCloud(object);
  voxelgrid.filter(*downsampled);
  *object = *downsampled;
  cout<<"after"<<object->points.size()<<endl;

  // Estimate normals for scene
  bool DONE = false;
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;
  nest.setRadiusSearch (0.4);
  nest.setInputCloud (object);
  nest.compute (*object);

  list<int> idx_cloud;
  vector<int> normalNum(object->points.size(),0);
  for(size_t i=0; i<object->points.size(); ++i){
    idx_cloud.push_back(i);
  }

  list<int>::iterator iter = idx_cloud.begin();
  while(!idx_cloud.empty()){
    int id = *iter;
    Eigen::Vector3f v1(object->points[*iter].normal_x,object->points[*iter].normal_y,object->points[*iter].normal_z);
    v1.normalize();
    iter = idx_cloud.erase(iter);
    for(; iter != idx_cloud.end();){
      Eigen::Vector3f v2(object->points[*iter].normal_x,object->points[*iter].normal_y,object->points[*iter].normal_z);
      v2.normalize();
      float tmp = v1.dot(v2);
      if(fabs(tmp-1)<0.004){
        normalNum[id]+=1;
        iter = idx_cloud.erase(iter);
      }else{
        iter++;
      }
    }
    iter = idx_cloud.begin();
  }
  int maxNormal = findMaxNormal(normalNum);
  Eigen::Vector3f groundNormal(object->points[maxNormal].normal_x,object->points[maxNormal].normal_y,object->points[maxNormal].normal_z);
  groundNormal.normalize();
  PointCloudNT::Ptr object_max (new PointCloudNT);
  object_max->points.push_back(object->points[maxNormal]);

  //add cube
  pcl::visualization::PCLVisualizer visu("Alignment");
  visu.addCoordinateSystem(5.0);
  visu.addPointCloud (object, ColorHandlerT (object, 0.0, 255.0, 0.0), "object");
  visu.addPointCloudNormals<PointNT, PointNT> (object_max, object_max, 1, 10.8, "normals_max");
  for (int i=1; i<cloudVec.size(); ++i)
  {
    if(!cloudVec[i].empty()){
      PointCloudNT::Ptr object1 = cloudVec[i].makeShared();
      visu.addPointCloud (object1, ColorHandlerT (object1, 255.0, 0.0, 0.0), to_string(i));

      pcl::MomentOfInertiaEstimation <PointNT> feature_extractor;
      feature_extractor.setInputCloud (object1);
      feature_extractor.compute ();
      std::vector <float> moment_of_inertia;
      std::vector <float> eccentricity;
      PointNT min_point_OBB;
      PointNT max_point_OBB;
      PointNT position_OBB;
      Eigen::Matrix3f rotational_matrix_OBB;
      float major_value, middle_value, minor_value;
      Eigen::Vector3f major_vector, middle_vector, minor_vector;
      Eigen::Vector3f mass_center;

      feature_extractor.getMomentOfInertia (moment_of_inertia);
      feature_extractor.getEccentricity (eccentricity);
      feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
      feature_extractor.getEigenValues (major_value, middle_value, minor_value);
      feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
      feature_extractor.getMassCenter (mass_center);

      Eigen::Vector3f find_x = major_vector;
      if(fabs(middle_vector(0))>fabs(find_x(0)))
        find_x = middle_vector;
      if(fabs(minor_vector(0))>fabs(find_x(0)))
        find_x = minor_vector;
      find_x.normalize();
      float rotate_to_x = acos(find_x(0));
      if(rotate_to_x > (M_PI/2)){
        rotate_to_x = - (M_PI - rotate_to_x);
      }
      find_x(2)>=0 ? rotate_to_x : rotate_to_x *=(-1);

      Eigen::Vector3f find_y = major_vector;
      if(fabs(middle_vector(2))>fabs(find_y(2)))
        find_y = middle_vector;
      if(fabs(minor_vector(2))>fabs(find_y(2)))
        find_y = minor_vector;
      find_y.normalize();
      Eigen::Vector3f d_to_ground = Eigen::Vector3f(0,find_y(1),find_y(2)).cross(Eigen::Vector3f(0,groundNormal(1),groundNormal(2)));
      float rotate_to_ground = acos(find_y(1)*groundNormal(1)+find_y(2)*groundNormal(2));
      if(fabs(rotate_to_ground) > (M_PI/2)){
        rotate_to_ground = -(M_PI - rotate_to_ground);
      }
      find_y(2)<0 ? rotate_to_ground : rotate_to_ground *=(-1);
      cout<<rotate_to_ground<<"!!!!!!!!!"<<endl;
//      if(d_to_ground(0)<0)
//        rotate_to_ground*=(-1);

      Eigen::Matrix3f R;
      R = Eigen::AngleAxisf(rotate_to_x, Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(rotate_to_ground, Eigen::Vector3f::UnitX());
      rotational_matrix_OBB = R * rotational_matrix_OBB;

      Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
      Eigen::Quaternionf quat (rotational_matrix_OBB);
      visu.addCube (position, quat, max_point_OBB.x - min_point_OBB.x,
                    max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB"+to_string(i));
      visu.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                       pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB"+to_string(i));

      pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
      pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1),
                            major_vector (2) + mass_center (2));
      pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1),
                            middle_vector (2) + mass_center (2));
      pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1),
                            minor_vector (2) + mass_center (2));
      visu.addLine (center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector"+to_string(i));
      visu.addLine (center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector"+to_string(i));
      visu.addLine (center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector"+to_string(i));
    }
  }

  while(!visu.wasStopped())
  {
    visu.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  return (0);
}
