cmake_minimum_required(VERSION 3.9)

IF (WIN32)
#path to opencv build directory
set(OpenCV_DIR D:/opencv/cv31/opencv/build)
set(OpenCV_INCLUDES ${OpenCV_DIR}/include)
endif(WIN32)

set(PROJECT example)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
enable_language(CUDA) 
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCES
	main.cpp
	sources/resizeGPU.cu
	sources/resizeCPU.cpp
	sources/resizeOMP.cpp
	sources/converter.cpp
)

include_directories(
	${OpenCV_INCLUDES}
	${CUDA_INCLUDES}
	include/
)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH TRUE)

add_executable(${PROJECT} ${SOURCES})

target_link_libraries(${PROJECT}
	${OpenCV_LIBS} 
	${CUDA_LIBRARIES}
)

IF (MSVC)
#set(CMAKE_CXX_FLAGS_RELEASE "/MT")
#set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
endif(MSVC)
