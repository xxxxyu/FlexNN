#ifndef DUMMY_MAT_H
#define DUMMY_MAT_H

#include "stdio.h"
#include "allocator.h"
#include "mat.h"

namespace flexnn {

class NCNN_EXPORT DummyMat
{
public:
    DummyMat();
    DummyMat(const DummyMat& m);
    DummyMat(int _w, size_t _elemsize = 4UL, ncnn::Allocator* _allocator = nullptr);
    DummyMat(int _w, int _h, size_t _elemsize = 4UL, ncnn::Allocator* _allocator = nullptr);
    DummyMat(int _w, int _h, int _c, size_t _elemsize = 4UL, ncnn::Allocator* _allocator = nullptr);
    DummyMat(int _w, int _h, int _d, int _c, size_t _elemsize = 4UL, ncnn::Allocator* _allocator = nullptr);
    DummyMat(const ncnn::Mat& m);

    ~DummyMat();

    int create(int _w, size_t _elemsize = 4UL, ncnn::Allocator* /*_allocator*/ = nullptr);
    int create(int _w, int _h, size_t _elemsize = 4UL, ncnn::Allocator* /*_allocator*/ = nullptr);
    int create(int _w, int _h, int _c, size_t _elemsize = 4UL, ncnn::Allocator* /*_allocator*/ = nullptr);
    int create(int _w, int _h, int _d, int _c, size_t _elemsize = 4UL, ncnn::Allocator* /*_allocator*/ = nullptr);

    // reshape vec
    DummyMat reshape(int w, ncnn::Allocator* allocator = 0) const;
    // reshape image
    DummyMat reshape(int w, int h, ncnn::Allocator* allocator = 0) const;
    // reshape dim
    DummyMat reshape(int w, int h, int c, ncnn::Allocator* allocator = 0) const;
    // reshape cube
    DummyMat reshape(int w, int h, int d, int c, ncnn::Allocator* allocator = 0) const;

    void create_like(const DummyMat& m, ncnn::Allocator* _allocator);

    DummyMat clone(ncnn::Allocator* /*_allocator*/ = nullptr) const;

    const DummyMat channel_range(int _c, int channels) const;
    DummyMat channel_range(int _c, int channels);

    DummyMat& operator=(const DummyMat& m);

    void addref();

    void release();

    bool empty() const;

    size_t total() const;

public:
    int dims;
    int w;
    int h;
    int d;
    int c;
    size_t cstep;

    int* refcount;
    size_t elemsize;  // bytes
    size_t totalsize; // bytes

    ncnn::Allocator* allocator;
};

void copy_make_border(const flexnn::DummyMat& src, DummyMat& dst, int top, int bottom, int left, int right, int type, float v, const ncnn::Option& opt);

} // namespace flexnn

#endif // DUMMY_MAT_H
