#include <torch/extension.h>
#include <THC/THC.h>
#include <iostream>
#include <vector>

extern THCState* state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

at::Tensor generate_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor idxs        // [hn,vn,2]);
);

at::Tensor generate_hypothesis(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor idxs        // [hn,vn,2]
)
{
    CHECK_INPUT(direct);
    CHECK_INPUT(coords);
    CHECK_INPUT(idxs);

    return generate_hypothesis_launcher(direct, coords, idxs);
}

void voting_for_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor hypo_pts,   // [hn,vn,3]
    at::Tensor inliers,    // [hn,vn,tn]
    float inlier_thresh
);

void voting_for_hypothesis(
    at::Tensor direct,     // [tn,vn,3]
    at::Tensor coords,     // [tn,3]
    at::Tensor hypo_pts,   // [hn,vn,3]
    at::Tensor inliers,    // [hn,vn,tn]
    float inlier_thresh
)
{
    CHECK_INPUT(direct);
    CHECK_INPUT(coords);
    CHECK_INPUT(hypo_pts);
    CHECK_INPUT(inliers);

    voting_for_hypothesis_launcher(direct, coords, hypo_pts, inliers, inlier_thresh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_hypothesis", &generate_hypothesis, "generate hypothesis");
    m.def("voting_for_hypothesis", &voting_for_hypothesis, "voting for hypothesis");
}
