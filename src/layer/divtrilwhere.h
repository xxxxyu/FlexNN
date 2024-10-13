#ifndef DIV_TRIL_WHERE_H
#define DIV_TRIL_WHERE_H

#include "layer.h"

namespace ncnn {

class DivTrilWhere : public Layer
{
public:
    DivTrilWhere();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // DIV_TRIL_WHERE_H