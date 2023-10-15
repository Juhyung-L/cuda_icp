#pragma once
#include <vector>

#include "cuda_runtime.h"
#include "cusolverDn.h"

#include "cuda_icp/octree.hpp"
#include "cuda_icp/rviz_visualizer.hpp"

class ScanMatcher
{
public:
    ScanMatcher(RvizVisualizer& vis);
    ~ScanMatcher();
    void nearestNeighbor(double3* d_pc, int* min_dists_idx);
    void matchScan(std::vector<double3>& src_pc, Octree& tgt_octree);
private:
    void setUpSVD();

    int THREADS_PER_BLOCK = 256; // arbitrarily selected

    // svd variables
    cublasHandle_t cublas_hdl;
    cusolverDnHandle_t cusolver_hdl;
    cudaStream_t stream;

    // all variables involved in calculation
    const int m = 3;
    const int n = 3;

    OctNode* d_tgt_octree;
    double3* u_tgt_pc;
    double3* u_src_pc;
    
    double3* u_corr_src_pc;
    double3* u_corr_tgt_pc;

    int* u_min_dists_idx;

    double* u_R;
    double3* u_T;
    double* u_error_arr;

    double* d_W_arr;
    double* d_W;
    double* d_S;
    double* d_U;
    double* d_VT;
    int* devInfo;

    int lwork = 0;
    double* d_work;
    double* d_rwork;

    double error_threshold = 0.0000005;
    int max_iter = 50;

    // visualizer
    RvizVisualizer vis_;
};