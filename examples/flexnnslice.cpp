#include "flexnnslice.h"

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
    if (argc < 6)
    {
        fprintf(stderr, "usage: %s <inparam> <inbin> <outparam> <outbin> <flag> [<conv_sz> <fc_sz>]\n", argv[0]);
        return -1;
    }

    double start = flexnn::get_current_time();

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    int flag = atoi(argv[5]);
    const char* cutstartname = nullptr;
    const char* cutendname = nullptr;

    int max_fc_size = 5e7;
    int max_conv_size = 5e7;

    if (argc >= 7)
    {
        max_conv_size = atoi(argv[6]) / 4;
    }
    if (argc >= 8)
    {
        max_fc_size = atoi(argv[7]) / 4;
    }

    FlexnnSlice slicer;

    if (flag == 65536 || flag == 1)
    {
        slicer.storage_type = 1;
    }
    else
    {
        slicer.storage_type = 0;
    }

    // fprintf(stderr, "load model %s %s\n", inparam, inbin);

    slicer.load_param_dummy(inparam);

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        slicer.load_model(dr);
        slicer.gen_random_weight = true;
    }
    else
    {
        int ret = slicer.load_model(inbin);
        if (ret)
        {
            // fallback to random
            DataReaderFromEmpty dr;
            slicer.load_model(dr);
            slicer.gen_random_weight = true;
        }
    }

    // resolve all shapes at first
    slicer.shape_inference();

    // graph modification
    slicer.slice_innerproduct(max_fc_size);
    slicer.slice_convolution(max_conv_size);

    // topological sort
    slicer.topological_sort();

    // resolve shapes again
    slicer.shape_inference();

    // pre-transform
    slicer.transform_kernel_convolution(max_conv_size);

    slicer.save(outparam, outbin);

    double end = flexnn::get_current_time();
    fprintf(stderr, "total slicing time: %.2f ms\n", end - start);

    return 0;
}