cmake_minimum_required(VERSION 3.3)

# Find ros dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ov_core REQUIRED)

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
        ${CERES_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/ceres/Factor_GenericPrior.cpp
        src/ceres/Factor_ImageReprojCalib.cpp
        src/ceres/Factor_ImuCPIv1.cpp
        src/ceres/State_JPLQuatLocal.cpp
        src/init/InertialInitializer.cpp
        src/dynamic/DynamicInitializer.cpp
        src/static/StaticInitializer.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_init_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
ament_target_dependencies(ov_init_lib rclcpp ov_core cv_bridge)
target_link_libraries(ov_init_lib ${thirdparty_libraries})
target_include_directories(ov_init_lib PUBLIC src/)
install(TARGETS ov_init_lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_init_lib)

##################################################
# Make binary files!
##################################################




# Install launch and config directories
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)

# finally define this as the package
ament_package()

