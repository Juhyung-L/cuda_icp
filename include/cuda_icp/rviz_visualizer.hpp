#pragma once

#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "cuda_runtime.h"

class RvizVisualizer
{
public:
    RvizVisualizer(rclcpp::Node::SharedPtr node) : node_(node)
    {
        marker_pub = node_->create_publisher<visualization_msgs::msg::Marker>(
            "marker", 10
        );
        marker_array_pub = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
            "marker_array", 10
        );

        node_->declare_parameter("point_size", 0.001);
        node_->declare_parameter("line_size", 0.001);
    }

    void visualizePC(const double3* points, size_t n, double r, double g, double b, bool delete_last=false)
    {
        double point_size = node_->get_parameter("point_size").as_double();

        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";

        if (last_pc_id != 0 && delete_last) // delete the previous point cloud
        {
            m.action = visualization_msgs::msg::Marker::DELETE;
            m.id = last_pc_id;
            marker_pub->publish(m);
        }
        m.header.stamp = node_->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.scale.x = point_size;
        m.scale.y = point_size;
        m.scale.z = point_size;
        // make sure alpha and color is set or else the points will be invisible
        std_msgs::msg::ColorRGBA c;
        c.a = 1.0;
        c.r = r;
        c.g = g;
        c.b = b;
        m.color = c;
        if (delete_last)
        {
            last_pc_id = node_->now().nanoseconds(); // store the marker id for deletion later
            m.id = last_pc_id;
        }
        else
        {
            m.id = node_->now().nanoseconds();
            last_pc_id = 0;
        }
        m.action = visualization_msgs::msg::Marker::ADD;
        geometry_msgs::msg::Point p;
        for (size_t i=0; i<n; ++i)
        {
            p.x = points[i].x;
            p.y = points[i].y;
            p.z = points[i].z;

            m.points.push_back(p);
        }

        marker_pub->publish(m);
    }

    void traverseOctree(int node_idx, OctNode* octree, visualization_msgs::msg::MarkerArray& cube_array)
    {
        OctNode cur_node = octree[node_idx];
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = node_->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.pose.position.x = cur_node.center.x;
        m.pose.position.y = cur_node.center.y;
        m.pose.position.z = cur_node.center.z;
        m.scale.x = cur_node.half_length * 2.f;
        m.scale.y = cur_node.half_length * 2.f;
        m.scale.z = cur_node.half_length * 2.f;
        m.color.a = 0.1;
        m.color.b = 1.0;
        m.id = node_idx;
        m.action = visualization_msgs::msg::Marker::ADD;
        cube_array.markers.push_back(m);

        if (cur_node.is_leaf)
        {
            return;
        }
        else
        {
            for (int i=0; i<8; ++i)
            {
                traverseOctree(cur_node.first_child_idx + i, octree, cube_array);
            }
        }
    }

    void visualizeOctree(Octree& octree)
    {
        visualization_msgs::msg::MarkerArray cube_array;
        traverseOctree(0, octree.node_pool, cube_array);
        marker_array_pub->publish(cube_array);
    }

    void visualizeNN(double3* corr_src_pc, double3* corr_tgt_pc, int n, double r, double g, double b, bool delete_last)
    {
        double line_size = node_->get_parameter("line_size").as_double();

        // visualization_msgs::msg::MarkerArray line_array;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = node_->now();
        if (last_line_list_id != 0 && delete_last) // delete the previous point cloud
        {
            m.action = visualization_msgs::msg::Marker::DELETE;
            m.id = last_line_list_id;
            marker_pub->publish(m);
        }
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.scale.x = line_size;
        std_msgs::msg::ColorRGBA c;
        c.a = 1.0;
        c.r = r;
        c.g = g;
        c.b = b;
        m.color = c;
        if (delete_last)
        {
            last_line_list_id = node_->now().nanoseconds(); // store the marker id for deletion later
            m.id = last_line_list_id;
        }
        else
        {
            m.id = node_->now().nanoseconds();
            last_line_list_id = 0;
        }
        m.action = visualization_msgs::msg::Marker::ADD;
        geometry_msgs::msg::Point p;
        for (int i=0; i<n; ++i)
        {
            p.x = corr_src_pc[i].x;
            p.y = corr_src_pc[i].y;
            p.z = corr_src_pc[i].z;
            m.points.push_back(p);

            p.x = corr_tgt_pc[i].x;
            p.y = corr_tgt_pc[i].y;
            p.z = corr_tgt_pc[i].z;
            m.points.push_back(p);
        }
        marker_pub->publish(m);
    }
    
private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub;

    int last_pc_id = 0;
    int last_line_list_id = 0;
};