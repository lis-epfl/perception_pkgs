cmake_minimum_required(VERSION 3.3)

# Find ros dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    message(FATAL_ERROR "ROS 2 (ament) is required for this pruned package.")
endif ()
add_definitions(-DROS_AVAILABLE=2)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)


##################################################
# OV_NVJPG: the ISOLATED hardware-JPEG shim.
# Its own shared object on purpose.  This TU must never see /usr/include/jpeglib.h and must
# never see OpenCV: Jetson's libnvjpeg.so is a full libjpeg fork whose jpeg_common_fields
# macro inserts 14 members ahead of `client_data`, so every field after `progress` sits at a
# different offset.  An isolated target is what ENFORCES the include order; the estimator
# dlopen()s it RTLD_LOCAL|RTLD_DEEPBIND so neither side can see the other's jpeg_*.
##################################################
set(JMM /usr/src/jetson_multimedia_api)
if (EXISTS ${JMM}/samples/common/classes/NvJpegDecoder.cpp)
    add_library(ov_nvjpg SHARED
            src/track/nvjpg_shim.cpp
            ${JMM}/samples/common/classes/NvJpegDecoder.cpp
            ${JMM}/samples/common/classes/NvElement.cpp
            ${JMM}/samples/common/classes/NvElementProfiler.cpp
            ${JMM}/samples/common/classes/NvLogging.cpp
            ${JMM}/samples/common/classes/NvBuffer.cpp
            ${JMM}/samples/common/classes/NvBufSurface.cpp)
    # libjpeg-8b FIRST: that directory holds NVIDIA's jpeglib.h and it must win.
    target_include_directories(ov_nvjpg BEFORE PRIVATE
            ${JMM}/include/libjpeg-8b
            ${JMM}/include)
    target_compile_options(ov_nvjpg PRIVATE -fvisibility=hidden -O2 -Wno-unused-parameter
            -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable)
    target_link_libraries(ov_nvjpg PRIVATE
            -L/usr/lib/aarch64-linux-gnu/tegra -lnvjpeg -lnvbufsurface -lnvbufsurftransform -lpthread -ldl)
    install(TARGETS ov_nvjpg LIBRARY DESTINATION lib)
    message(STATUS "OV_NVJPG: building libov_nvjpg.so (isolated NVJPG shim)")
else ()
    message(STATUS "OV_NVJPG: jetson_multimedia_api samples absent -- shim NOT built")
endif ()

##################################################
# Make the core library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/clahe_cuda.cu
        src/track/gpu_track.cu
        src/track/nvjpg_decode.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
ament_target_dependencies(ov_core_lib rclcpp cv_bridge)
target_link_libraries(ov_core_lib ${thirdparty_libraries} cudart nvvpi
        -L/usr/lib/aarch64-linux-gnu/tegra -lnvbufsurface -lEGL)
target_include_directories(ov_core_lib PRIVATE ${JMM}/include /usr/local/cuda/include)
set_target_properties(ov_core_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ov_core_lib PUBLIC src/)
install(TARGETS ov_core_lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_core_lib)

##################################################
# Make binary files!
##################################################

# TODO: UPGRADE THIS TO ROS2 AS ANOTHER FILE!!
#if (catkin_FOUND AND ENABLE_ROS)
#    add_executable(test_tracking src/test_tracking.cpp)
#    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
#endif ()



# finally define this as the package
ament_package()