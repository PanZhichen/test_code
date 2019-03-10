cmake_minimum_required(VERSION 2.8.3)
project(process_points)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
    ndt_omp
)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED REQUIRED COMPONENTS io registration segmentation visualization)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(colors
		src/colors.cpp
		)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES process_points colors
  CATKIN_DEPENDS roscpp rospy std_msgs
  DEPENDS system_lib
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
#add_executable(ICP_test_node src/ICP_test.cpp)
#target_link_libraries(ICP_test_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
#
#add_executable(pub_points_node src/pub_points.cpp)
#target_link_libraries(pub_points_node
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        )
link_libraries(colors)
add_executable(save_bag_node src/save_bag.cpp)
target_link_libraries(save_bag_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
add_executable(ICP_pcd_node src/ICP_pcd.cpp)
target_link_libraries(ICP_pcd_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
#add_executable(fpfh_node src/fpfh_test.cpp src/NDT_test.cpp)
#target_link_libraries(fpfh_node
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        )
add_executable(keypoint_node src/keypoint.cpp)
target_link_libraries(keypoint_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
add_executable(NARF_node src/NARF_Ros.cpp)
target_link_libraries(NARF_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
add_executable(NDT_node src/NDT_test.cpp)
target_link_libraries(NDT_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
add_executable(cube_node src/cube.cpp)
target_link_libraries(cube_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
	colors
        )
