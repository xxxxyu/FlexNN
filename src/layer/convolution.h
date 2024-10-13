// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"

namespace ncnn {

class Convolution : public Layer
{
public:
    Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int load_model(const ModelBin& mb, const Option& opt);

    virtual int release_model();

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& top_blob, const Option& opt) const;

    void make_padding(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& bottom_blob_bordered) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

    void make_padding(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

#if NCNN_INT8
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_INT8
    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    Mat top_blob_int8_scales;
#endif

    // weight data type (transformed or not) and layout.
    // 0 = origin flattened (default ncnn), 1 = origin CHW, 2 = im2col gemm CHW, 3 = winograd63 CHW, 4 = winograd43 CHW, 5 = winograd23 CHW, 6 = conv3x3s2 CHW.
    int weight_data_type;
    int weight_w;
    int weight_h;
    int weight_c;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_H
