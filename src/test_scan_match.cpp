#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "cuda_runtime.h"

#include "cuda_icp/scan_matcher.hpp"
#include "cuda_icp/octree.hpp"
#include "cuda_icp/transform.hpp"
#include "cuda_icp/rviz_visualizer.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("scanmatch_test");
    auto vis = RvizVisualizer(node);

    // path to point cloud file
    node->declare_parameter("file_path", rclcpp::ParameterType::PARAMETER_STRING); // throw error if uninitialized

    // parameters for point cloud
    node->declare_parameter("rx", 0.0);
    node->declare_parameter("ry", 0.0);
    node->declare_parameter("rz", 0.0);
    node->declare_parameter("tx", 0.0);
    node->declare_parameter("ty", 0.0);
    node->declare_parameter("tz", 0.0);

    // parameters for octree
    node->declare_parameter("max_tree_depth", 4);
    node->declare_parameter("half_length", 1.0);
    node->declare_parameter("centerx", 0.0);
    node->declare_parameter("centery", 0.0);
    node->declare_parameter("centerz", 0.0);

    // parse data file
    std::ifstream pc_data_file(node->get_parameter("file_path").as_string());
    
    std::vector<double3> tgt_pc;
    
    std::string line;
    double3 point;
    while (std::getline(pc_data_file, line))
    {
        std::stringstream ss(line);
        ss >> point.x;
        ss >> point.y;
        ss >> point.z;

        tgt_pc.push_back(point);
    }
    
    // make the octree for the untransformed point cloud
    // this is the target point cloud
    Octree octree;
    double3 center;
    center.x = node->get_parameter("centerx").as_double();
    center.y = node->get_parameter("centery").as_double();
    center.z = node->get_parameter("centerz").as_double();
    octree.createOctree(tgt_pc, center, node->get_parameter("half_length").as_double(), node->get_parameter("max_tree_depth").as_int());
    // vis.visualizeOctree(octree);
    
    vis.visualizePC(tgt_pc.data(), tgt_pc.size(), 1.0, 0.0, 0.0);

    std::vector<double3> src_pc(tgt_pc);

    // transform the point cloud
    // this is the input point cloud that will be transformed into the target point cloud
    double r[3] = {node->get_parameter("rx").as_double(), 
                   node->get_parameter("ry").as_double(), 
                   node->get_parameter("rz").as_double()};
    double t[3] = {node->get_parameter("tx").as_double(), 
                   node->get_parameter("ty").as_double(), 
                   node->get_parameter("tz").as_double()};
    transform3DPCGPU(r[0], r[1], r[2], t[0], t[1], t[2], src_pc);
    
    ScanMatcher scan_matcher(vis);
    scan_matcher.matchScan(src_pc, octree);
    // vis.visualizePC(src_pc.data(), src_pc.size(), 0.0, 1.0, 0.0);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}