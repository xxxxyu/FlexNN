#include "gather.h"
#include <cmath>

namespace ncnn {

Gather::Gather()
{
    one_blob_only = false;
}

int Gather::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int w = bottom_blobs[1].w;
    int n_embd = bottom_blobs[0].w;

    ncnn::Mat& top_blob = top_blobs[0];
    top_blob.create(n_embd, w, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    float* dst = top_blob;
    const float* in = bottom_blobs[1];
    const float* weight = bottom_blobs[0];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int c = 0; c < w; c++)
    {
        int idx = std::round(*in) * n_embd;
        memcpy(dst, weight + idx, n_embd * 4);
        in++;
        dst += n_embd;
    }

    return 0;
}

int Gather::forward(const std::vector<flexnn::DummyMat>& bottom_blobs, std::vector<flexnn::DummyMat>& top_blobs, const Option& opt) const
{
    int w = bottom_blobs[1].w;
    int n_embd = bottom_blobs[0].w;

    flexnn::DummyMat& top_blob = top_blobs[0];
    top_blob.create(n_embd, w, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    return 0;
}

} // namespace ncnn