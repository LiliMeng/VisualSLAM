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

SET(LAPACKBLAS_DIR "/usr/lib" CACHE PATH "Path to lapack/blas libraries")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(include/DBoW2/ ${OpenCV_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS})
include_directories(include/DLoopDetector/ ${OpenCV_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
include_directories(include/libfovis/ )


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIR})


LINK_DIRECTORIES(${LAPACKBLAS_DIR})
SET(HAVE_LAPACK 1 CACHE BOOL "Do we have LAPACK/BLAS?")
# the directory where the lapack/blas/f2c libraries reside
SET(LAPACKBLAS_DIR "/usr/lib" CACHE PATH "Path to lapack/blas libraries")
SET(NEED_F2C 1 CACHE BOOL "Do we need either f2c or F77/I77?")
SET(HAVE_PLASMA 0 CACHE BOOL "Do we have PLASMA parallel linear algebra library?")
IF(HAVE_PLASMA)
 SET(PLASMA_DIR "/usr/local/PLASMA" CACHE PATH "Path to PLASMA root")
ENDIF(HAVE_PLASMA)
SET(LINSOLVERS_RETAIN_MEMORY 1 CACHE BOOL "Should linear solvers retain working memory between calls? (non-reentrant!)")
SET(LM_DBL_PREC 1 CACHE BOOL "Build double precision routines?")
SET(LM_SNGL_PREC 1 CACHE BOOL "Build single precision routines?")
OPTION(BUILD_DEMO "Build demo program?" TRUE)

# actual names for the lapack/blas/f2c libraries
SET(LAPACKBLAS_LIB_NAMES "lapack;blas" CACHE STRING "The name of the lapack & blas libraries")
#SET(LAPACKBLAS_LIB_NAMES "mkl_solver_sequential;mkl_intel_c;mkl_sequential;mkl_core" CACHE STRING "The name of the lapack library") # MKL
IF(NEED_F2C)
  SET(F2C_LIB_NAME f2c CACHE STRING "The name of the f2c or F77/I77 library")
  # f2c is sometimes equivalent to libF77 & libI77
  #SET(F2C_LIB_NAME "libF77;libI77" CACHE STRING "The name of the f2c or F77/I77 library")
ELSE(NEED_F2C)
  SET(F2C_LIB_NAME "" CACHE STRING "The name of the f2c or F77/I77 library")
ENDIF(NEED_F2C)

# actual names for the PLASMA libraries
IF(HAVE_PLASMA)
  SET(PLASMA_LIB_NAMES "plasma;coreblas;quark;lapacke" CACHE STRING "The names of the PLASMA libraries")
ENDIF(HAVE_PLASMA)


IF(0)
# treat paths relative to the source dir.
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0015 NEW)
ENDIF(COMMAND CMAKE_POLICY)
ENDIF(0)

# compiler flags
#ADD_DEFINITIONS(-DLINSOLVERS_RETAIN_MEMORY) # do not free memory between linear solvers calls
#REMOVE_DEFINITIONS(-DLINSOLVERS_RETAIN_MEMORY) # free memory between calls

########################## NO CHANGES BEYOND THIS POINT ##########################

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/levmar.h.in ${CMAKE_CURRENT_SOURCE_DIR}/levmar.h)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
IF(HAVE_PLASMA)
 INCLUDE_DIRECTORIES(${PLASMA_DIR}/include ${PLASMA_DIR}/quark)
ENDIF(HAVE_PLASMA)

# PLASMA headers in Axb.c should be compiled as C++
IF(HAVE_PLASMA)
  SET_SOURCE_FILES_PROPERTIES(Axb.c PROPERTIES LANGUAGE CXX)
ENDIF(HAVE_PLASMA)

# levmar library source files
ADD_LIBRARY(levmar STATIC
  lm.c Axb.c misc.c lmlec.c lmbc.c lmblec.c lmbleic.c
  levmar.h misc.h compiler.h
)




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
		      ${DBoW2_LIBS}
		      ${DLib_LIBS}
                      DLib
                      DBoW2
                     # DLoopDetector
		      f2c
                      levmar
                      blas
                      m
                      lapack
                      lcm
                    #  DVision
                    #  DUtilsCV
                    #  DUtils
                      isam 
                      amd 
                      camd 
                      colamd 
                      ccolamd 
                      cxsparse 
                      cholmod)
