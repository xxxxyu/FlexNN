#include "dummymat.h"
#include "layer.h"
#include "layer_type.h"

namespace flexnn {

DummyMat::DummyMat()
    : dims(0), w(0), h(0), d(0), c(0), cstep(0), refcount(nullptr), elemsize(0), allocator(nullptr)
{
}
DummyMat::DummyMat(const DummyMat& m)
    : dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c), cstep(m.cstep), refcount(m.refcount), elemsize(m.elemsize), allocator(m.allocator)
{
    addref();
}
DummyMat::DummyMat(int _w, size_t _elemsize, ncnn::Allocator* _allocator)
    : dims(0), w(0), h(0), d(0), c(0), cstep(0), refcount(nullptr), elemsize(0), allocator(nullptr)
{
    create(_w, _elemsize, _allocator);
}
DummyMat::DummyMat(int _w, int _h, size_t _elemsize, ncnn::Allocator* _allocator)
    : dims(0), w(0), h(0), d(0), c(0), cstep(0), refcount(nullptr), elemsize(0), allocator(nullptr)
{
    create(_w, _h, _elemsize, _allocator);
}
DummyMat::DummyMat(int _w, int _h, int _c, size_t _elemsize, ncnn::Allocator* _allocator)
    : dims(0), w(0), h(0), d(0), c(0), cstep(0), refcount(nullptr), elemsize(0), allocator(nullptr)
{
    create(_w, _h, _c, _elemsize, _allocator);
}
DummyMat::DummyMat(int _w, int _h, int _d, int _c, size_t _elemsize, ncnn::Allocator* _allocator)
    : dims(0), w(0), h(0), d(0), c(0), cstep(0), refcount(nullptr), elemsize(0), allocator(nullptr)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

void mat_to_dummy(const ncnn::Mat& src, DummyMat& dst)
{
    if (src.dims == 1)
        dst.create(src.w, src.elemsize);
    else if (src.dims == 2)
        dst.create(src.w, src.h, src.elemsize);
    else if (src.dims == 3)
        dst.create(src.w, src.h, src.c, src.elemsize);
    else if (src.dims == 4)
        dst.create(src.w, src.h, src.d, src.c, src.elemsize);
}

DummyMat::DummyMat(const ncnn::Mat& m)
{
    mat_to_dummy(m, *this);
}

DummyMat::~DummyMat()
{
    release();
    if (refcount)
    {
        fprintf(stderr, "refcount is not deleted!");
        delete refcount;
        refcount = nullptr;
    }
}

int DummyMat::create(int _w, size_t _elemsize, ncnn::Allocator* /*_allocator*/)
{
    release();
    dims = 1;
    w = _w;
    h = 1;
    c = 1;
    d = 1;
    elemsize = _elemsize;
    cstep = w;
    totalsize = ncnn::alignSize(total() * elemsize, 4);
    refcount = new int;
    *refcount = 1;
    return 0;
}
int DummyMat::create(int _w, int _h, size_t _elemsize, ncnn::Allocator* /*_allocator*/)
{
    release();
    dims = 2;
    w = _w;
    h = _h;
    c = 1;
    d = 1;
    elemsize = _elemsize;
    cstep = (size_t)w * h;
    totalsize = ncnn::alignSize(total() * elemsize, 4);
    refcount = new int;
    *refcount = 1;
    return 0;
}
int DummyMat::create(int _w, int _h, int _c, size_t _elemsize, ncnn::Allocator* /*_allocator*/)
{
    release();
    dims = 3;
    w = _w;
    h = _h;
    c = _c;
    d = 1;
    elemsize = _elemsize;
    cstep = ncnn::alignSize((size_t)w * h * elemsize, 16) / elemsize;
    totalsize = ncnn::alignSize(total() * elemsize, 4);
    refcount = new int;
    *refcount = 1;
    return 0;
}
int DummyMat::create(int _w, int _h, int _d, int _c, size_t _elemsize, ncnn::Allocator* /*_allocator*/)
{
    release();
    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;
    elemsize = _elemsize;
    cstep = ncnn::alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
    totalsize = ncnn::alignSize(total() * elemsize, 4);
    refcount = new int;
    *refcount = 1;
    return 0;
}

DummyMat DummyMat::reshape(int _w, ncnn::Allocator* _allocator) const
{
    if (w * h * d * c != _w)
        return DummyMat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        DummyMat m;
        m.create(_w, elemsize, _allocator);

        return m;
    }

    DummyMat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

DummyMat DummyMat::reshape(int _w, int _h, ncnn::Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h)
        return DummyMat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        DummyMat m;
        m.create(_w, _h, elemsize, _allocator);

        return m;
    }

    DummyMat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;

    m.cstep = (size_t)_w * _h;

    return m;
}

DummyMat DummyMat::reshape(int _w, int _h, int _c, ncnn::Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _c)
        return DummyMat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != ncnn::alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            DummyMat m;
            m.create(_w, _h, _c, elemsize, _allocator);

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        DummyMat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    DummyMat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;

    m.cstep = ncnn::alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

DummyMat DummyMat::reshape(int _w, int _h, int _d, int _c, ncnn::Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return DummyMat();

    if (dims < 3)
    {
        if ((size_t)_w * _h * _d != ncnn::alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        {
            DummyMat m;
            m.create(_w, _h, _d, _c, elemsize, _allocator);

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        DummyMat tmp = reshape(_w * _h * _d * _c, _allocator);
        return tmp.reshape(_w, _h, _d, _c, _allocator);
    }

    DummyMat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;

    m.cstep = ncnn::alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;

    return m;
}

void DummyMat::create_like(const DummyMat& m, ncnn::Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, _allocator);
}

DummyMat DummyMat::clone(ncnn::Allocator* /*_allocator*/) const
{
    if (empty())
        return DummyMat();

    DummyMat m;
    if (dims == 1)
        m.create(w, elemsize);
    else if (dims == 2)
        m.create(w, h, elemsize);
    else if (dims == 3)
        m.create(w, h, c, elemsize);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize);

    return m;
}

DummyMat DummyMat::channel_range(int /*_c*/, int channels)
{
    DummyMat m(w, h, d, channels, elemsize, allocator);
    m.dims = dims;
    return m;
}

const DummyMat DummyMat::channel_range(int /*_c*/, int channels) const
{
    DummyMat m(w, h, d, channels, elemsize, allocator);
    m.dims = dims;
    return m;
}

DummyMat& DummyMat::operator=(const DummyMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    refcount = m.refcount;
    elemsize = m.elemsize;
    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

void DummyMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

void DummyMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        delete refcount;
    }

    elemsize = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;
    d = 0;

    cstep = 0;
    refcount = nullptr;
}

bool DummyMat::empty() const
{
    return dims == 0 || refcount == nullptr;
}

size_t DummyMat::total() const
{
    return cstep * c;
}

void copy_make_border(const flexnn::DummyMat& src, DummyMat& dst, int top, int bottom, int left, int right, int type, float v, const ncnn::Option& opt)
{
    ncnn::Layer* padding = ncnn::create_layer(ncnn::LayerType::Padding);

    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    // padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    // padding->destroy_pipeline(opt);

    delete padding;
}

} // namespace flexnn