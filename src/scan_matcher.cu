#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"

#include "cuda_icp/scan_matcher.hpp"
#include "cuda_icp/transform.hpp"
#include "cuda_icp/error_handle.h"

const double DOUBLE_MAX = std::numeric_limits<double>::max();

ScanMatcher::ScanMatcher(RvizVisualizer& vis) : vis_(vis)
{
    // cuSolver and cuBlas setup
    cusolverCheck(cusolverDnCreate(&cusolver_hdl));
    cublasCheck(cublasCreate(&cublas_hdl));
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverCheck(cusolverDnSetStream(cusolver_hdl, stream));

    setUpSVD();
}

void ScanMatcher::setUpSVD()
{
    // cudaMallocManaged for arrays that is accessed in both CPU and GPU
    cudaCheck(cudaMallocManaged(&d_W, m*n*sizeof(double)));
    cudaCheck(cudaMallocManaged(&d_S, n*sizeof(double)));
    cudaCheck(cudaMalloc(&d_U, m*m*sizeof(double)));
    cudaCheck(cudaMalloc(&d_VT, m*n*sizeof(double)));
    cudaCheck(cudaMallocManaged(&devInfo, sizeof(int)));

    cusolverCheck(cusolverDnDgesvd_bufferSize(cusolver_hdl, m, n, &lwork));
    cudaCheck(cudaMalloc(&d_work, lwork*sizeof(double)));
}

ScanMatcher::~ScanMatcher()
{
    cudaCheck(cudaFree(d_W));
    cudaCheck(cudaFree(d_S));
    cudaCheck(cudaFree(d_U));
    cudaCheck(cudaFree(d_VT));
    cudaCheck(cudaFree(devInfo));
    cudaCheck(cudaFree(d_work));
}

__global__ void kernNearestNeighbor(const double3* src_pc, const double3* tgt_pc, OctNode* tgt_octree, int* min_dists_idx, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        double min_dist = DOUBLE_MAX;
        int min_dist_idx = -1;

        // find the leaf octant that contains the input point
        int node_idx = 0;
        while (!tgt_octree[node_idx].is_leaf)
        {
            bool x = src_pc[idx].x > tgt_octree[node_idx].center.x;
            bool y = src_pc[idx].y > tgt_octree[node_idx].center.y;
            bool z = src_pc[idx].z > tgt_octree[node_idx].center.z;

            node_idx = tgt_octree[node_idx].first_child_idx + (x + 2*y + 4*z);
        }
        
        OctNode &leaf_node = tgt_octree[node_idx];
        // get the minimum distance between input point and all the points in the leaf octant
        for (int i=0; i<tgt_octree[node_idx].num_points; ++i)
        {
            // no need to square since we just need to know which point is the closest
            double dist_squared = pow(src_pc[idx].x - tgt_pc[leaf_node.points_idx[i]].x, 2) + 
                                  pow(src_pc[idx].y - tgt_pc[leaf_node.points_idx[i]].y, 2) + 
                                  pow(src_pc[idx].z - tgt_pc[leaf_node.points_idx[i]].z, 2);
            if (dist_squared < min_dist)
            {
                min_dist = dist_squared;
                min_dist_idx = leaf_node.points_idx[i];
            }
        }
        min_dists_idx[idx] = min_dist_idx;
    }
}

__global__ void kernComputeNorm(double3* pc, const double3 centroid, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        pc[idx].x -= centroid.x;
        pc[idx].y -= centroid.y;
        pc[idx].z -= centroid.z;
    }
}

__global__ void kernComputeWArray(const double3* src_norm, const double3* tgt_norm, double* W_arr, const int w, const int h, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        W_arr[idx*w*h+0] = tgt_norm[idx].x * src_norm[idx].x; 
        W_arr[idx*w*h+1] = tgt_norm[idx].y * src_norm[idx].x;
        W_arr[idx*w*h+2] = tgt_norm[idx].z * src_norm[idx].x;

        W_arr[idx*w*h+3] = tgt_norm[idx].x * src_norm[idx].y; 
        W_arr[idx*w*h+4] = tgt_norm[idx].y * src_norm[idx].y;
        W_arr[idx*w*h+5] = tgt_norm[idx].z * src_norm[idx].y;

        W_arr[idx*w*h+6] = tgt_norm[idx].x * src_norm[idx].z; 
        W_arr[idx*w*h+7] = tgt_norm[idx].y * src_norm[idx].z;
        W_arr[idx*w*h+8] = tgt_norm[idx].z * src_norm[idx].z;
    }

}

__global__ void kernComputeW(const double* W_arr, double* W, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i=idx; i<N*blockDim.x; i += blockDim.x)
    {
        W[idx] += W_arr[i];
    }
}

__global__ void kernComputeErrorArray(double* error_arr, const double3* src_pc, const double3* tgt_pc, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N)
    {
        error_arr[idx] = pow(src_pc[idx].x - tgt_pc[idx].x, 2) + pow(src_pc[idx].y - tgt_pc[idx].y, 2) + pow(src_pc[idx].z - tgt_pc[idx].z, 2);
    }
}

void ScanMatcher::matchScan(std::vector<double3>& src_pc, Octree& tgt_octree)
{
    size_t NUM_SRC_PC = src_pc.size();
    size_t NUM_TGT_PC = tgt_octree.point_cloud.size();

    // gpu memory allocation
    cudaCheck(cudaMallocManaged(&u_src_pc, NUM_SRC_PC*sizeof(double3)));
    cudaCheck(cudaMemcpy(u_src_pc, src_pc.data(), NUM_SRC_PC*sizeof(double3), cudaMemcpyDefault));
    cudaCheck(cudaMallocManaged(&u_tgt_pc, NUM_TGT_PC*sizeof(double3)));
    cudaCheck(cudaMemcpy(u_tgt_pc, tgt_octree.point_cloud.data(), NUM_TGT_PC*sizeof(double3), cudaMemcpyDefault));
    cudaCheck(cudaMalloc(&d_tgt_octree, tgt_octree.num_nodes*sizeof(OctNode)));
    cudaCheck(cudaMemcpy(d_tgt_octree, tgt_octree.node_pool, tgt_octree.num_nodes*sizeof(OctNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMallocManaged(&u_min_dists_idx, NUM_SRC_PC*sizeof(int)));
    cudaCheck(cudaMallocManaged(&u_corr_src_pc, NUM_SRC_PC*sizeof(double3)));
    cudaCheck(cudaMallocManaged(&u_corr_tgt_pc, NUM_SRC_PC*sizeof(double3)));
    cudaCheck(cudaMalloc(&d_W_arr, NUM_SRC_PC*m*n*sizeof(double)));
    cudaCheck(cudaMallocManaged(&u_R, m*n*sizeof(double)));
    cudaCheck(cudaMallocManaged(&u_T, sizeof(double3)));
    cudaCheck(cudaMallocManaged(&u_error_arr, NUM_SRC_PC*sizeof(double)));

    double e = DOUBLE_MAX;
    int iter = 0;

    while(iter < max_iter && e > error_threshold)
    {
        // reset W matrix
        cudaCheck(cudaMemset(d_W, 0, m*n*sizeof(double)));

        // find nearest neighbor point in tgt_pc for each point in src_pc
        // some points might not find a nearest neighbor
        kernNearestNeighbor<<<(NUM_SRC_PC+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_src_pc, u_tgt_pc, d_tgt_octree, u_min_dists_idx, NUM_SRC_PC
        );
        cudaCheck(cudaDeviceSynchronize());

        int NUM_CORR = 0;
        // make the correspondence pc
        for (size_t i=0; i<NUM_SRC_PC; ++i)
        {
            if (u_min_dists_idx[i] != -1) // ignore index -1 (means nearest neighbor could not be found)
            {
                u_corr_src_pc[NUM_CORR].x = u_src_pc[i].x;
                u_corr_src_pc[NUM_CORR].y = u_src_pc[i].y;
                u_corr_src_pc[NUM_CORR].z = u_src_pc[i].z;

                u_corr_tgt_pc[NUM_CORR].x = u_tgt_pc[u_min_dists_idx[i]].x;
                u_corr_tgt_pc[NUM_CORR].y = u_tgt_pc[u_min_dists_idx[i]].y;
                u_corr_tgt_pc[NUM_CORR].z = u_tgt_pc[u_min_dists_idx[i]].z;
                ++NUM_CORR;
            }
        }

        // visualize
        vis_.visualizePC(u_src_pc, NUM_SRC_PC, 0.0, 1.0, 0.0, true);
        // vis_.visualizeNN(u_corr_src_pc, u_corr_tgt_pc, NUM_CORR, 0.0, 0.0, 1.0, true);

        // calculate centriods for src and tgt correspondence pc
        double3 src_centroid = make_double3(0.0, 0.0, 0.0);
        double3 tgt_centroid = make_double3(0.0, 0.0, 0.0);
        for (int i=0; i<NUM_CORR; ++i)
        {
            src_centroid.x += u_corr_src_pc[i].x;
            src_centroid.y += u_corr_src_pc[i].y;
            src_centroid.z += u_corr_src_pc[i].z;

            tgt_centroid.x += u_corr_tgt_pc[i].x;
            tgt_centroid.y += u_corr_tgt_pc[i].y;
            tgt_centroid.z += u_corr_tgt_pc[i].z;
        }
        src_centroid.x /= NUM_CORR;
        src_centroid.y /= NUM_CORR;
        src_centroid.z /= NUM_CORR;

        tgt_centroid.x /= NUM_CORR;
        tgt_centroid.y /= NUM_CORR;
        tgt_centroid.z /= NUM_CORR;

        // subtract the centroid from every point
        kernComputeNorm<<<(NUM_CORR+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_corr_src_pc, src_centroid, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());
        kernComputeNorm<<<(NUM_CORR+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_corr_tgt_pc, tgt_centroid, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());

        // calcualte W matrix
        kernComputeWArray<<<(NUM_CORR+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_corr_src_pc, u_corr_tgt_pc, d_W_arr, m, n, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());
        kernComputeW<<<1,m*n>>>(
            d_W_arr, d_W, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());

        // perform SVD
        cusolverCheck(
            cusolverDnDgesvd(cusolver_hdl, 'A', 'A', m, n, d_W, m, d_S, d_U,
                m, d_VT, m, d_work, lwork, d_rwork, devInfo)
        );
        if ((*devInfo) != 0)
        {
            fprintf(stderr, "SVD failed\n");
            break;
        }
        cudaCheck(cudaDeviceSynchronize());

        double alpha = 1.0;
        double beta = 0.0;
        // calculate the optimal rotation matrix
        cublasCheck(cublasDgemm(cublas_hdl, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, m, &alpha, d_U, m, d_VT, m, &beta, u_R, m
        ));
        cudaCheck(cudaDeviceSynchronize());

        // calculate the optimal translation matrix
        (*u_T).x =  tgt_centroid.x - (u_R[0]*src_centroid.x + u_R[3]*src_centroid.y + u_R[6]*src_centroid.z);
        (*u_T).y =  tgt_centroid.y - (u_R[1]*src_centroid.x + u_R[4]*src_centroid.y + u_R[7]*src_centroid.z);
        (*u_T).z =  tgt_centroid.z - (u_R[2]*src_centroid.x + u_R[5]*src_centroid.y + u_R[8]*src_centroid.z);
        
        // apply the rotation and translation to src_pc
        kernTransform3DPC<<<(NUM_SRC_PC+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_src_pc, u_R, u_T, NUM_SRC_PC
        );
        cudaCheck(cudaDeviceSynchronize());

        // update error
        // apply the transformation to src correspondence pc too because error is calculated from it
        kernTransform3DPC<<<(NUM_CORR+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_corr_src_pc, u_R, u_T, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());
        kernComputeErrorArray<<<(NUM_CORR+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            u_error_arr, u_corr_src_pc, u_corr_tgt_pc, NUM_CORR
        );
        cudaCheck(cudaDeviceSynchronize());

        e = 0.0;
        for (int i=0; i<NUM_CORR; ++i)
        {
            e += u_error_arr[i];
        }
        e /= NUM_CORR;
        fprintf(stderr, "Error: %.10f\n", e); // print 10 decimal places
        ++iter;
        fprintf(stderr, "Iteration: %d\n", iter);
        
        // for debug
        // char a = getchar();
        // if(a == 'q')
        // {
        //     fprintf(stderr, "Exiting scan matching loop\n");
        //     break;
        // }

        // for recording
        // sleep(1);
        // usleep(50000);
    }

    // copy the matched source point cloud
    src_pc.clear();
    for (size_t i=0; i<NUM_SRC_PC; ++i)
    {
        src_pc.push_back(u_src_pc[i]);
    }

    cudaCheck(cudaFree(u_src_pc));
    cudaCheck(cudaFree(u_tgt_pc));
    cudaCheck(cudaFree(d_tgt_octree));
    cudaCheck(cudaFree(u_min_dists_idx));
    cudaCheck(cudaFree(u_corr_src_pc));
    cudaCheck(cudaFree(u_corr_tgt_pc));
    cudaCheck(cudaFree(d_W_arr));
    cudaCheck(cudaFree(u_R));
    cudaCheck(cudaFree(u_T));
    cudaCheck(cudaFree(u_error_arr));
}

