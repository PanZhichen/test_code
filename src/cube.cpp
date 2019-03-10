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
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>

using namespace std;
// Types
typedef pcl::PointXYZINormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloudTI;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ColorHandlerTI;
typedef pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> ColorHandlerTIG;

struct CloudWithLabel{
    int label;
    pcl::PointCloud<pcl::PointXYZI> cloud;
};
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
int scanFiles(vector<string> &fileList, string inputDirectory)
{
  inputDirectory = inputDirectory.append("/");

  DIR *p_dir;
  const char* str = inputDirectory.c_str();

  p_dir = opendir(str);
  if( p_dir == NULL)
  {
    cout<< "can't open :" << inputDirectory << endl;
    return 0;
  }

  struct dirent *p_dirent;

  while ( p_dirent = readdir(p_dir))
  {
    string tmpFileName = p_dirent->d_name;
    if( tmpFileName == "." || tmpFileName == "..")
    {
      continue;
    }
    else
    {
      string endS = tmpFileName.substr(tmpFileName.size()-8,8);
      if(endS != "pred.txt"){
        fileList.push_back(inputDirectory+tmpFileName);
      }
    }
  }
  closedir(p_dir);
  return fileList.size();
}

int main (int argc, char **argv)
{
  // Point clouds
  PointCloudNT groundCloud;
  vector<PointCloudNT> cloudVec;
  vector<CloudWithLabel> cloudwithlabel;
  cloudVec.resize(300);
  // Get input object and scene
  if (argc != 2)
  {
    pcl::console::print_error ("Syntax is: %s object.pcd\n", argv[0]);
    return (1);
  }

  pcl::visualization::PCLVisualizer visu("Alignment");
  visu.addCoordinateSystem(5.0);
  Eigen::Vector3f groundNormal;
  bool INITED = false;
  vector<string> allfiles;
  ifstream in;
  int fileNum = scanFiles(allfiles,argv[1]);
  for(int i=0; i<allfiles.size()-1; ++i){
    for(int j=i+1; j<allfiles.size(); ++j){
      if(allfiles[j]<allfiles[i]){
        string tmp = allfiles[i];
        allfiles[i] = allfiles[j];
        allfiles[j] = tmp;
      }
    }
  }
  if(fileNum > 0){
    for(int i=0; i<fileNum; ++i){
//      size_t lengthS = allfiles[i].size();
//      string endS = allfiles[i].substr(lengthS-8,8);
//      if(endS != "pred.txt"){
//        cout<<allfiles[i]<<endl;
//        continue;
        cloudwithlabel.clear();
        groundCloud.clear();
        //读取文件数据
        in.open(allfiles[i], ios::in);
        if (! in.is_open())
        { cout << "Error opening file"; exit (1); }
        string buff;
        while(getline(in, buff)){
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
          if(nums[4] > 0){
            pcl::PointXYZI point;
            point.x = nums[0];
            point.y = nums[1];
            point.z = nums[2];
            point.intensity = nums[3];
            bool exist = false;
            for(int i=0; i<cloudwithlabel.size(); ++i){
              if(cloudwithlabel[i].label == nums[4]){
                exist = true;
                cloudwithlabel[i].cloud.points.push_back(point);
              }
            }
            if(!exist){
              CloudWithLabel tmp;
              tmp.label = nums[4];
              tmp.cloud.points.push_back(point);
              cloudwithlabel.push_back(tmp);
            }
          }else{
//            if(!INITED){
              PointNT groundP;
              groundP.x = nums[0];
              groundP.y = nums[1];
              groundP.z = nums[2];
              groundP.intensity = nums[3];
              groundCloud.points.push_back(groundP);
//            }
          }
        }
        in.close();

        PointCloudNT::Ptr object = groundCloud.makeShared();
//        cout<<"before"<<object->points.size()<<endl;
        PointCloudNT::Ptr downsampled(new PointCloudNT);
        pcl::VoxelGrid<PointNT> voxelgrid;
        voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
        voxelgrid.setInputCloud(object);
        voxelgrid.filter(*downsampled);
        *object = *downsampled;
//        cout<<"after"<<object->points.size()<<endl;
        //估计地面法向量
        if(!INITED){
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
          groundNormal = Eigen::Vector3f(object->points[maxNormal].normal_x,object->points[maxNormal].normal_y,object->points[maxNormal].normal_z);
          groundNormal.normalize();
//          visu.addPointCloud (object, ColorHandlerT (object, 0.0, 255.0, 0.0), "ground");
//          visu.addPointCloud (object, pcl::visualization::PointCloudColorHandlerGenericField<PointNT>(object, "intensity"), "ground");
//          visu.addPointCloudNormals<PointNT, PointNT> (object_max, object_max, 1, 10.8, "normals_max");
          INITED = true;
        }
        visu.removeAllPointClouds();
        visu.removeAllShapes();
        visu.addPointCloud (groundCloud.makeShared(),
                pcl::visualization::PointCloudColorHandlerGenericField<PointNT>(groundCloud.makeShared(), "intensity"), "ground");

        //add cube
        for (int i=0; i<cloudwithlabel.size(); ++i){
          if(!cloudwithlabel[i].cloud.empty()){
            PointCloudTI::Ptr object1 = cloudwithlabel[i].cloud.makeShared();
//            visu.addPointCloud (object1, ColorHandlerTI (object1, 255.0, 0.0, 0.0), to_string(i));
            visu.addPointCloud (object1, ColorHandlerTIG (object1, "intensity"), to_string(i));
            pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
            feature_extractor.setInputCloud (object1);
            feature_extractor.compute ();
            std::vector <float> moment_of_inertia;
            std::vector <float> eccentricity;
            pcl::PointXYZI min_point_OBB;
            pcl::PointXYZI max_point_OBB;
            pcl::PointXYZI position_OBB;
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
            Eigen::Vector3f d_to_x = Eigen::Vector3f(find_x(0),0,find_x(2)).cross(Eigen::Vector3f(1,0,0));
            float rotate_to_x = acos(find_x(0));
            if(rotate_to_x > (M_PI/2)){
              rotate_to_x = -(M_PI - rotate_to_x);
            }
            if(d_to_x(1)<0)
              rotate_to_x*=(-1);
//      if(fabs(d_to_x(0))<0.05)
//        rotate_to_x=0;
            //find_x(2)>=0 ? rotate_to_x : rotate_to_x *=(-1);

            Eigen::Vector3f find_z = major_vector;
            if(fabs(middle_vector(2))>fabs(find_z(2)))
              find_z = middle_vector;
            if(fabs(minor_vector(2))>fabs(find_z(2)))
              find_z = minor_vector;
            find_z.normalize();
//      cout<<find_z<<"!!!!!!"<<endl;
            Eigen::Vector3f d_to_z = Eigen::Vector3f(0,find_z(1),find_z(2)).cross(Eigen::Vector3f(0,groundNormal(1),groundNormal(2)));
//      cout<<d_to_z<<"!!!~~!!!"<<endl;
            float rotate_to_z = acos(find_z(1)*groundNormal(1) + find_z(2)*groundNormal(2));
            if(fabs(rotate_to_z) > (M_PI/2)){
              rotate_to_z = -(M_PI - rotate_to_z);
            }
            if(d_to_z(0)<0)
              rotate_to_z*=(-1);
            if(fabs(d_to_z(0))<0.1)
              rotate_to_z=0;
//      cout<<rotate_to_z<<"!!!!!!!!!"<<endl;
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            R = Eigen::AngleAxisf(rotate_to_x, Eigen::Vector3f::UnitY())*
                Eigen::AngleAxisf(rotate_to_z, Eigen::Vector3f::UnitX());
//      cout<<"ang:="<<R.eulerAngles(0,1,2)<<"!!!!!!!!!"<<endl;
            rotational_matrix_OBB = R* rotational_matrix_OBB;

            Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
            Eigen::Quaternionf quat (rotational_matrix_OBB);
            visu.addCube (position, quat, max_point_OBB.x - min_point_OBB.x,
                          max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB"+to_string(i));
//            visu.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
//                                             pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB"+to_string(i));
            visu.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                             255.0,0.0,255.0, "OBB"+to_string(i));

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
//      }
      visu.spinOnce (100);
    }
    return 0;
  }else{
    return(1);
  }
}
