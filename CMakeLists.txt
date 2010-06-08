cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULES_PATH} ${CMAKE_SOURCE_DIR}/Modules)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Modules/cuda" ${CMAKE_MODULE_PATH})

SET(PROJECT_NAME "gicu")

include(MacroEnsureOutOfSourceBuild)
macro_ensure_out_of_source_build("${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there.")


PROJECT(${PROJECT_NAME})

#SET(CMAKE_CXX_FLAGS "-g")
#SET(CMAKE_CXX_FLAGS_DEBUG)

SET( ${PROJECT_NAME}_SRCS
     gicu.cpp )

find_package(GTK2 COMPONENTS gtk)
FIND_PACKAGE(CUDA)


INCLUDE_DIRECTORIES(
	${GTK2_INCLUDE_DIRS}
 	${CURL_INCLUDE_DIR}
	/usr/include/gimp-2.0
	/usr/include/glib-2.0
	/usr/lib64/glib-2.0/include
	/opt/cuda/include
	/opt/cuda/sdk/C/common/inc
)

LINK_DIRECTORIES(
	/opt/cuda/lib64
	/opt/cuda/sdk/C/lib
)

include_directories(${CUDA_INCLUDE_DIRS})


CUDA_ADD_EXECUTABLE( ${PROJECT_NAME}
  gicu.cpp
  cuda.cu
)

#ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${GTK2_LIBRARIES} cutil cudart gimp-2.0)

