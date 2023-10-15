cmake_minimum_required(VERSION 3.8)
project(cuda_icp CUDA CXX C)

set(CMAKE_BUILD_TYPE Release) #None, Debug, Release, RelWithDebInfo, MinSizeRel
set(CMAKE_CXX_STANDARD 17)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra)
# endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) # prints warning on build if not defined
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(dependencies
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
)

add_library(libtransform STATIC src/transform.cu)
add_library(liboctree STATIC src/octree.cu)
add_library(libscan_matcher STATIC src/scan_matcher.cu)
target_link_libraries(libscan_matcher CUDA::cusolver CUDA::cublas)
ament_target_dependencies(libscan_matcher ${dependencies})

add_executable(test_scan_match src/test_scan_match.cpp)
ament_target_dependencies(test_scan_match ${dependencies})
target_link_libraries(test_scan_match libtransform liboctree libscan_matcher)

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

install(TARGETS
  test_scan_match
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
