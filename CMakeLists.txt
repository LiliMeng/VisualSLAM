cmake_minimum_required(VERSION 2.6.0)

project(SLAM)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PCL 1.6 REQUIRED)


find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(DLib REQUIRED)
find_package(DLoopDetector REQUIRED)
find_package(libfovis REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(include/DBoW2/ ${OpenCV_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS})
include_directories(include/DLoopDetector/ ${OpenCV_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
include_directories(include/libfovis/ )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIR})


add_definitions(-Wall)

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3")

FILE(GLOB srcs *.cpp)
FILE(GLOB odom_srcs Odometry/*.cpp)
FILE(GLOB dvos Odometry/dvo/*.cpp)
FILE(GLOB pr PlaceRecognition/*.cpp)
FILE(GLOB motion PlaceRecognition/MotionEstimation/*.cpp)
FILE(GLOB lcm PlaceRecognition/MotionEstimation/lcmtypes*.c)
FILE(GLOB pg PoseGraph/*.cpp)

source_group("Source Files" FILES ${srcs})
source_group("Odometry Source Files" FILES ${odom_srcs})
source_group("DVO Source Files" FILES ${dvos})
source_group("PlaceRecognition Source Files" FILES ${pr})
source_group("MotionEstimation Source Files" FILES ${motion})
source_group("LCM Source Files" FILES ${lcm})
source_group("PoseGraph Source Files" FILES ${pg})



add_executable(SLAM 
               ${srcs}
               ${odom_srcs}
               ${dvos}
               ${pr}
               ${motion}
               ${lcm}
               ${pg}
)

target_link_libraries(SLAM
                      ${ZLIB_LIBRARY}
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${PCL_LIBRARIES}
                      tbb
                      fovis
                      DLib
                      DBoW2
                      DLoopDetector
                      levmar
                      blas
                      m
                      lapack
                      lcm
                      DVision
                      DUtilsCV
                      DUtils
                      isam 
                      amd 
                      camd 
                      colamd 
                      ccolamd 
                      cxsparse 
                      cholmod)
