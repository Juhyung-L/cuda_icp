#include <cmath>
#include "cuda_icp/octree.hpp"

Octree::Octree()
{}

Octree::~Octree()
{
    free(node_pool);
}

void Octree::createOctree(std::vector<double3>& input_point_cloud, double3& center, double half_length, int max_tree_depth)
{
    point_cloud = input_point_cloud;
    max_num_nodes = (pow(8, max_tree_depth+1)-1)/7;
    node_pool = (OctNode*)malloc(max_num_nodes*sizeof(OctNode));
    OctNode root_node;
    root_node.first_child_idx = 0;
    root_node.center = center;
    root_node.half_length = half_length;
    root_node.is_leaf = true;
    root_node.num_points = 0;
    node_pool[0] = root_node;
    num_nodes = 1;

    for (size_t i=0; i<point_cloud.size(); ++i)
    {
        insert(0, i);
    }
}

void Octree::insert(int node_idx, int point_idx)
{
    OctNode &cur_node = node_pool[node_idx];
    if (cur_node.is_leaf) // leaf node
    {
        if (cur_node.num_points < MAX_PTS_PER_OCTANT) // node is not full
        {
            cur_node.points_idx[cur_node.num_points] = point_idx;
            cur_node.num_points++;
        }
        else // node is full
        {
            // subdivide the node
            cur_node.first_child_idx = num_nodes;
            cur_node.is_leaf = false;
            double new_half_length = cur_node.half_length / 2.0;

            for (int z=0; z<2; ++z)
            {
                for (int y=0; y<2; ++y)
                {
                    for (int x=0; x<2; ++x)
                    {
                        OctNode child_node;

                        // set the new center
                        child_node.center.x = cur_node.center.x + new_half_length * (x ? 1:-1);
                        child_node.center.y = cur_node.center.y + new_half_length * (y ? 1:-1);
                        child_node.center.z = cur_node.center.z + new_half_length * (z ? 1:-1);
                        
                        child_node.num_points = 0;
                        child_node.half_length = new_half_length;
                        child_node.is_leaf = true;
                        node_pool[num_nodes++] = child_node;
                    }
                }
            }

            // redistribute the points
            for (size_t i=0; i<cur_node.num_points; ++i)
            {
                insert(node_idx, cur_node.points_idx[i]);
            }
            // insert the input point into the parent node
            insert(node_idx, point_idx);
        }
    }
    else // not a leaf node - continue going down the tree
    {
        bool x = point_cloud[point_idx].x > cur_node.center.x;
        bool y = point_cloud[point_idx].y > cur_node.center.y;
        bool z = point_cloud[point_idx].z > cur_node.center.z;

        int new_node_idx = cur_node.first_child_idx + (x + 2*y + 4*z);

        insert(new_node_idx, point_idx);
    }
}

