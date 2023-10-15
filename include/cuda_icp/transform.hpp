#pragma once
#include <vector>
#include "cuda_runtime.h"

__global__ void kernTransform3DPC(double3* points, double* R, double3* T, int n);

void transform3DPCGPU(double rx, double ry, double rz,
                      double tx, double ty, double tz,
                      std::vector<double3>& points);