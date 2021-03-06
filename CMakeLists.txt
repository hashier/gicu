CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULES_PATH} ${CMAKE_SOURCE_DIR}/Modules)
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Modules/cuda" ${CMAKE_MODULE_PATH})

SET(PROJECT_NAME "gicu")

INCLUDE(MacroEnsureOutOfSourceBuild)
macro_ensure_out_of_source_build("${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there.")

PROJECT(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS "-Os")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")

SET( ${PROJECT_NAME}_SRCS
	gicu.cpp )

#GCC 4.3
#SET(CUDA_NVCC_FLAGS -keep "-use_fast_math;-arch=compute_12;-code=sm_12;-ccbin;/usr/bin/g++-4.3")
#SET(CUDA_NVCC_FLAGS "--compiler-bindir=/usr/bin/gcc-4.3.4")
#SET(CUDA_NVCC_FLAGS "--compiler-bindir=/usr/bin/gcc-4.3")


FIND_PACKAGE(GTK2 COMPONENTS gtk)
FIND_PACKAGE(CUDA)

IF ( NOT CUDA_TOOLKIT_ROOT_DIR)
	SET( CUDA_TOOLKIT_ROOT_DIR
		/afs/tu-bs.de/usr/usr53/y0053765/cuda/cuda
		)
ENDIF ( NOT CUDA_TOOLKIT_ROOT_DIR)

INCLUDE_DIRECTORIES(
	${GTK2_INCLUDE_DIRS}
	${CURL_INCLUDE_DIR}
	/usr/include/gimp-2.0
	/usr/include/glib-2.0

	/usr/lib64/glib-2.0/include

	/opt/cuda/sdk/C/common/inc

	/afs/tu-bs.de/usr/usr53/y0053765/cuda/cuda/sdk/C/common/inc
)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

CUDA_INCLUDE_DIRECTORIES (
	/opt/cuda/sdk/C/common/inc
	)

LINK_DIRECTORIES(
	/opt/cuda/lib64
	/opt/cuda/sdk/C/lib
)

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME}
	gicu.cpp
	cuda.cu
	gui.cpp
	#grey.cu
	#box.cu
	#sobel.cu
)

#ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${GTK2_LIBRARIES} ${CUDA_LIBRARIES} gimp-2.0 gimpui-2.0)

#INSTALL(FILES ${files} DESTINATION include)
INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gicu DESTINATION ~/.gimp-2.6/plug-ins/)

# ON IBR MACHINES!!!!!!
# ON IBR MACHINES!!!!!!
# ON IBR MACHINES!!!!!!
# ON IBR MACHINES!!!!!!
# ON IBR MACHINES!!!!!!
# run cmake and afterwards you have run ccmake and change the variable
# CUDA_NVCC_FLAGS to:
# --compiler-bindir=/usr/bin/gcc-4.3
MESSAGE(STATUS "Please compile with gcc 4.3.x... see CMakeLists.txt for more information")
MESSAGE(STATUS "Please compile with gcc 4.3.x... see CMakeLists.txt for more information")
MESSAGE(STATUS "Please compile with gcc 4.3.x... see CMakeLists.txt for more information")
MESSAGE(STATUS "Please compile with gcc 4.3.x... see CMakeLists.txt for more information")
