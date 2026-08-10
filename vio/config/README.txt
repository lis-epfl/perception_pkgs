Intentionally almost empty.

Upstream OpenVINS ships ~8 MB of public-dataset configs here; none are used (this package
passes ../../configs/*.yaml by absolute path). The DIRECTORY must still exist:
ov_msckf/cmake/ROS2.cmake:111 runs install(DIRECTORY ../config/ ...) and colcon fails
without it. Do not delete this directory or this file.
