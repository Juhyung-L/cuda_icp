#pragma once
#include <vector>
#include "cuda_runtime.h"
#define MAX_PTS_PER_OCTANT 1000

struct OctNode
{
    int first_child_idx;
    int points_idx[MAX_PTS_PER_OCTANT];
    double3 center;
    double half_length;
    bool is_leaf;
    int num_points;
};

class Octree
{
public:
    Octree();
    ~Octree();
    void createOctree(std::vector<double3>& input_point_cloud, double3& center, double half_length, int max_tree_depth);
    
    OctNode* node_pool;
    int num_nodes;
    int max_num_nodes;
    std::vector<double3> point_cloud;
private:
    void insert(int node_idx, int point_idx);
};