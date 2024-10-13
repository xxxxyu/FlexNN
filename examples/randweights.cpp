#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <stack>
#include <queue>

// ncnn public header
#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "layer_type.h"

// ncnn private header
#include "modelwriter.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <param> <bin>\n", argv[0]);
        return -1;
    }

    const char* param = argv[1];
    const char* bin = argv[2];

    ModelWriter mw;

    DataReaderFromEmpty dr;
    mw.load_param(param);
    mw.load_model(dr);
    mw.gen_random_weight = true;

    mw.save(param, bin);

    return 0;
}