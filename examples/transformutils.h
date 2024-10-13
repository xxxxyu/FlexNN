#ifndef TRANSFORM_UTILS_H
#define TRANSFORM_UTILS_H

// ncnn public header
#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "layer_type.h"

#include "cpu.h"

#include "layer/arm/arm_usability.h"

namespace ncnn {

// ncnn kernel implementation
#include "layer/arm/convolution_3x3_winograd.h"
#include "layer/arm/convolution_im2col_gemm.h"
#include "layer/arm/convolution_3x3.h"

static void convolution_im2col_gemm_transform_kernel_wrapper(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    convolution_im2col_gemm_transform_kernel(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
}

static void conv3x3s1_winograd43_transform_kernel_wrapper(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd43_transform_kernel(kernel, AT, inch, outch, opt);
}

static void conv3x3s1_winograd63_transform_kernel_wrapper(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd63_transform_kernel(kernel, AT, inch, outch, opt);
}
} // namespace ncnn

#endif // TRANSFORM_UTILS_H