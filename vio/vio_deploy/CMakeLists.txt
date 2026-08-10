cmake_minimum_required(VERSION 3.5)
project(vio_deploy NONE)

find_package(ament_cmake REQUIRED)

# Flight-mode estimator config + calibration-prior stiffness env file.
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config/)

# run_serial.sh is the ONLY supported way to invoke the estimator; build_openvins.sh
# builds this workspace. Installed executable so they can be found via the ament index.
install(PROGRAMS scripts/run_serial.sh scripts/build_openvins.sh
        DESTINATION lib/${PROJECT_NAME})

ament_package()
