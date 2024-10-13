#ifndef GATHER_H
#define GATHER_H

#include "layer.h"

namespace ncnn {

class Gather : public Layer
{
public:
    Gather();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward(const std::vector<flexnn::DummyMat>& bottom_blobs, std::vector<flexnn::DummyMat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // GATHER_H