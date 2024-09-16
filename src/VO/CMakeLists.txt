cmake_minimum_required(VERSION 3.0.2)
project(VO)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  cv_bridge
  image_transport
  tf
  sensor_msgs
)
find_package(OpenCV  REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_library(cf_lib
 src/camera.cpp
 src/frame.cpp
 src/VO.cpp
 )
 target_link_libraries(
  cf_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES firt_slam
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  $(CMAKE_INCLUDE_CURRENT_DIR)/include/
  ${glog_INCLUDE_DIRS}
)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Ceres REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(VO src/VO.cpp)
target_link_libraries(VO cf_lib) 

