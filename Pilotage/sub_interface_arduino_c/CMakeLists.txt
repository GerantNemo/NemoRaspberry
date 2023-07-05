cmake_minimum_required(VERSION 3.8)
project(serial_test1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

#add_library(arduino_json SHARED include/serial_test1/ArduinoJson-v6.21.2.hpp)
add_library(arduino_json SHARED src/ArduinoJson-v6.21.2.cpp)
ament_target_dependencies(arduino_json)

add_executable(serial_node src/serial.cpp)
target_link_libraries(serial_node arduino_json)
ament_target_dependencies(serial_node rclcpp std_msgs geometry_msgs trajectory_msgs sensor_msgs)

install(TARGETS
    arduino_json
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS
  serial_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install resource files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

#Install include files
install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(
  include
)

ament_package()
