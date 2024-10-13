#include "divtrilwhere.h"

namespace ncnn {

DivTrilWhere::DivTrilWhere()
{
    one_blob_only = true;
}

int DivTrilWhere::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    top_blob.create(w, h, channels, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const float* src = bottom_blob.channel(p);
        float* dst = top_blob.channel(p);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (x > y)
                {
                    dst[0] = -1e4f;
                }
                else
                {
                    dst[0] = src[0] / 8.0f;
                }
                src++;
                dst++;
            }
        }
    }

    return 0;
}

int DivTrilWhere::forward(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    top_blob.create(w, h, channels, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    return 0;
}

} // namespace ncnn