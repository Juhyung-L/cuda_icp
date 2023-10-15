#include <cmath>
#include <vector>
#include "cuda_icp/transform.hpp"

__global__ void kernTransform3DPC(double3* points, double* R, double3* T, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        double x_cpy = points[i].x;
        double y_cpy = points[i].y;
        double z_cpy = points[i].z;
        points[i].x = x_cpy*R[0] + y_cpy*R[3] + z_cpy*R[6] + (*T).x;
        points[i].y = x_cpy*R[1] + y_cpy*R[4] + z_cpy*R[7] + (*T).y;
        points[i].z = x_cpy*R[2] + y_cpy*R[5] + z_cpy*R[8] + (*T).z;
    }
}

void transform3DPCGPU(double rx, double ry, double rz,
                      double tx, double ty, double tz,
                      std::vector<double3>& points)
{
    int num_points = points.size();
    
    // convert angles to radians
    rx = rx * M_PI / 180.f;
    ry = ry * M_PI / 180.f;
    rz = rz * M_PI / 180.f;

    // copy points to GPU   
    double3* d_points;
    cudaMalloc(&d_points, num_points*sizeof(double3));
    cudaMemcpy(d_points, points.data(), num_points*sizeof(double3), cudaMemcpyHostToDevice);
    
    // make the rotation matrix
    double* u_R;
    cudaMallocManaged(&u_R, 9*sizeof(double));

    // matrix formula is from https://en.wikipedia.org/wiki/Rotation_matrix
    u_R[0] = cos(ry)*cos(rz); u_R[3] = (sin(rx)*sin(ry)*cos(rz))-(cos(rx)*sin(rz)); u_R[6] = (cos(rx)*sin(ry)*cos(rz))+(sin(rx)*sin(rz));
    u_R[1] = cos(ry)*sin(rz); u_R[4] = (sin(rx)*sin(ry)*sin(rz))+(cos(rx)*cos(rz)); u_R[7] = (cos(rx)*sin(ry)*sin(rz))-(sin(rx)*cos(rz));
    u_R[2] = -sin(ry);        u_R[5] = sin(rx)*cos(ry);                             u_R[8] = cos(rx)*cos(ry);

    // make the translation vector
    double3* u_T;
    cudaMallocManaged(&u_T, sizeof(double3));
    (*u_T).x = tx;
    (*u_T).y = ty;
    (*u_T).z = tz;

    int num_threads = 256; // arbitrarly picked
    kernTransform3DPC<<<(num_points+num_threads-1)/num_threads, num_threads>>>(d_points, u_R, u_T, num_points);
    cudaDeviceSynchronize(); // blocks until all threads have completed

    // copy transformed point cloud to host
    cudaMemcpy(points.data(), d_points, num_points*sizeof(double3), cudaMemcpyDeviceToHost);

    // free up GPU memory
    cudaFree(d_points);
    cudaFree(u_R);
    cudaFree(u_T);
}