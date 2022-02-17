#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__
void generate_hypothesis_kernel(
    float* direct,     // [tn,vn,3]
    float* coords,     // [tn,3]
    int* idxs,         // [hn,vn,2]
    float* hypo_pts,   // [hn,vn,3]
    int tn,
    int vn,
    int hn
)
{
    int hvi = threadIdx.x + blockIdx.x*blockDim.x;
    if(hvi>=hn*vn) return;

    int hi=hvi/vn;
    int vi=hvi-hi*vn;

    int t0=idxs[hi*vn*2+vi*2];
    int t1=idxs[hi*vn*2+vi*2+1];

    float nx0=direct[t0*vn*3+vi*3];
    float ny0=direct[t0*vn*3+vi*3+1];
    float nz0=direct[t0*vn*3+vi*3+2];
    float cx0=coords[t0*3];
    float cy0=coords[t0*3+1];
    float cz0=coords[t0*3+2];

    float nx1=direct[t1*vn*3+vi*3];
    float ny1=direct[t1*vn*3+vi*3+1];
    float nz1=direct[t1*vn*3+vi*3+2];
    float cx1=coords[t1*3];
    float cy1=coords[t1*3+1];
    float cz1=coords[t1*3+2];

    // compute intersection
    if(fabs(ny0*nz1-nz0*ny1)<1e-6 && fabs(nz0*nx1-nx0*nz1)<1e-6 && fabs(nx0*ny1-ny0*nx1)<1e-6) return;
    float n00=nx0*nx0+ny0*ny0+nz0*nz0;
    float n11=nx1*nx1+ny1*ny1+nz1*nz1;
    float n01=nx0*nx1+ny0*ny1+nz0*nz1;
    float n0x10=nx0*(cx1-cx0)+ny0*(cy1-cy0)+nz0*(cz1-cz0);
    float n1x10=nx1*(cx1-cx0)+ny1*(cy1-cy0)+nz1*(cz1-cz0);
    float t=(n0x10*n11+n1x10*n01)/(n00*n11-n01*n01);
    float s=(n0x10*n01+n1x10*n00)/(n01*n01-n00*n11);
    float x=(cx0+cx1+t*nx0+s*nx1)/2.0;
    float y=(cy0+cy1+t*ny0+s*ny1)/2.0;
    float z=(cz0+cz1+t*nz0+s*nz1)/2.0;

    hypo_pts[hi*vn*3+vi*3]=x;
    hypo_pts[hi*vn*3+vi*3+1]=y;
    hypo_pts[hi*vn*3+vi*3+2]=z;
}

at::Tensor generate_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor idxs        // [hn,vn,2]
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=idxs.size(0);

    assert(direct.size(2)==3);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==3);
    assert(idxs.size(1)==vn);
    assert(idxs.size(2)==2);

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;

    getGPULayout(hn*vn, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);

    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    auto hypo_pts = at::zeros({hn, vn, 3}, direct.type());
    generate_hypothesis_kernel<<<bdim, tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        idxs.data<int>(),
        hypo_pts.data<float>(),
        tn, vn, hn
    );
    gpuErrchk(cudaGetLastError())

    return hypo_pts;
}

__global__
void voting_for_hypothesis_kernel(
    float* direct,     // [tn,vn,3]
    float* coords,     // [tn,3]
    float* hypo_pts,   // [hn,vn,3]
    unsigned char* inliers,     // [hn,vn,tn]
    int tn,
    int vn,
    int hn,
    float inlier_thresh
)
{
    int hi = threadIdx.x + blockIdx.x*blockDim.x;
    int vti = threadIdx.y + blockIdx.y*blockDim.y;
    if(hi>=hn||vti>=vn*tn) return;

    int vi=vti/tn;
    int ti=vti-vi*tn;

    float cx=coords[ti*3];
    float cy=coords[ti*3+1];
    float cz=coords[ti*3+2];

    float hx=hypo_pts[hi*vn*3+vi*3];
    float hy=hypo_pts[hi*vn*3+vi*3+1];
    float hz=hypo_pts[hi*vn*3+vi*3+2];

    float nx=direct[ti*vn*3+vi*3];
    float ny=direct[ti*vn*3+vi*3+1];
    float nz=direct[ti*vn*3+vi*3+2];

    float dx=hx-cx;
    float dy=hy-cy;
    float dz=hz-cz;

    float norm1=sqrt(nx*nx+ny*ny+nz*nz);
    float norm2=sqrt(dx*dx+dy*dy+dz*dz);
    if(norm1<1e-6||norm2<1e-6) return;

    float angle_dist=(dx*nx+dy*ny+dz*nz)/(norm1*norm2);
    if(angle_dist>inlier_thresh)
        inliers[hi*vn*tn+vi*tn+ti]=1;
}


void voting_for_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor hypo_pts,   // [hn,vn,3]
    at::Tensor inliers,    // [hn,vn,tn]
    float inlier_thresh
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=hypo_pts.size(0);

    assert(direct.size(2)==3);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==3);
    assert(hypo_pts.size(1)==vn);
    assert(hypo_pts.size(2)==3);
    assert(inliers.size(0)==hn);
    assert(inliers.size(1)==vn);
    assert(inliers.size(2)==tn);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn, vn*tn, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    voting_for_hypothesis_kernel<<<bdim,tdim>>>(
        direct.data<float>(),
        coords.data<float>(),
        hypo_pts.data<float>(),
        inliers.data<unsigned char>(),
        tn, vn, hn, inlier_thresh
    );
    gpuErrchk(cudaGetLastError())
}
