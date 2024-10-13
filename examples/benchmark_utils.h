#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <string.h>

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

// gpt2
#include <codecvt>
#include <locale>
#include <wchar.h>
#include <string>

int __Neg_Infinity = 0xFF800000;
const float Neg_Infinity = *((float*)&__Neg_Infinity);

inline void cooling_down(bool flag, int sec)
{
    if (flag)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(sec * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(sec);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = sec;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }
}

inline ncnn::Mat cstr2mat(const char* cstr) // parse c string of format "[d,c,h,w]" and return a ncnn::mat of the shape
{
    int w, h, c, d;
    sscanf(cstr, "[%d,%d,%d,%d]", &d, &c, &h, &w);

    ncnn::Mat m(w, h, d, c);
    return m;
}

inline bool matcmp(const ncnn::Mat& a, const ncnn::Mat& b, float delta = 1e-6) // returns false if 2 mats are identical
{
    fprintf(stderr, "Compare output mats: ");

    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.c != b.c || a.d != b.d)
    {
        fprintf(stderr, "dimensions not match.\n");
        return true;
    }

    int size = a.w * a.h * a.c * a.d;
    for (int i = 0; i < size; i++)
    {
        if (abs(a[i] - b[i]) >= delta)
        {
            fprintf(stderr, "%f != %f at %d.\n", a[i], b[i], i);
            return true;
        }
    }

    fprintf(stderr, "mats are identical.\n");
    return false;
}

inline void set_benchmark_config(ncnn::Option& opt, const char* config, int num_threads = 1) // TODO: allocators
{
    // common options
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.use_local_pool_allocator = false;
    opt.use_local_threads = true;

    // conv impl
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;

    // int8, fp16, packing
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = false;

    // config specific options
    if (strcmp(config, "ncnn_default") == 0)
    {
        // opt.blob_allocator = &g_blob_pool_allocator;
        // opt.workspace_allocator = &g_workspace_pool_allocator;
    }
    else if (strcmp(config, "ncnn_ondemand") == 0)
    {
        opt.use_ondemand_loading = true;
        // opt.use_winograd_convolution = true;
        // opt.use_sgemm_convolution = false;
    }
    else if (strcmp(config, "ncnn_parallel") == 0)
    {
        opt.use_parallel_preloading = true;
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = false;
    }
    else if (strcmp(config, "ncnn_direct_conv") == 0)
    {
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = false;
    }
    else if (strcmp(config, "flexnn_profile") == 0)
    {
        opt.use_ondemand_loading = true;
        opt.use_pretransform = true;
        opt.use_memory_profiler = true;
    }
    else if (strcmp(config, "flexnn_ondemand") == 0)
    {
        opt.use_ondemand_loading = true;
        opt.use_pretransform = true;
    }
    else if (strcmp(config, "flexnn_parallel") == 0)
    {
        opt.use_parallel_preloading = true;
        opt.use_pretransform = true;
    }
    else if (strcmp(config, "ncnn_ondemand_gemm") == 0)
    {
        opt.use_ondemand_loading = true;
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = true;
    }
    else if (strcmp(config, "ncnn_default_gemm") == 0)
    {
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = true;
    }
    else if (strcmp(config, "ncnn_ondemand_direct") == 0)
    {
        opt.use_ondemand_loading = true;
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = false;
    }
}

inline void load_layer_dependency(const char* path, std::vector<int>& layer_dependency)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to open %s\n", path);
        return;
    }

    char line[256];
    // int layer_index = 0;
    while (fgets(line, 256, fp))
    {
        int dependency_index;
        sscanf(line, "%d", &dependency_index);
        layer_dependency.push_back(dependency_index);
        // layer_index++;
    }

    fclose(fp);
}

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

// for gpt2
// ref: https://github.com/EdVince/GPT2-ChineseChat-NCNN

std::string WStringToString(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.to_bytes(wstr);
}

std::wstring StringToWString(const std::string& str)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.from_bytes(str);
}

std::vector<int> vector_merge(std::vector<int> v1, std::vector<int> v2)
{
    std::vector<int> v3;
    v3.insert(v3.end(), v1.begin(), v1.end());
    v3.insert(v3.end(), v2.begin(), v2.end());
    return v3;
}

void top_k_filtering(ncnn::Mat& logits)
{
    ncnn::Mat filtered_logits;
    filtered_logits.clone_from(logits);
    float* pt = filtered_logits;
    std::sort(pt, pt + 13317, std::greater<float>());
    float top_k_value = pt[8 - 1]; // topk的阈值
    for (int i = 0; i < 13317; i++)
    {
        if (logits[i] < top_k_value)
            logits[i] = Neg_Infinity;
    }
}

template<typename _Tp>
int softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i)
    {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

int multinomial(const ncnn::Mat& logits)
{
    ncnn::Mat weight;
    weight.clone_from(logits);
    for (int i = 1; i < 13317; i++)
        weight[i] += weight[i - 1];
    std::srand(static_cast<unsigned>(time(NULL)));
    float r = static_cast<float>(rand() % 13317) / 13317.0f;
    float* pt = weight;
    return std::lower_bound(pt, pt + 13317, r) - pt;
}

std::vector<int> token2idx(std::map<std::wstring, int>& tokenizer_token2idx, std::string token)
{
    std::vector<int> idx;
    std::wstring wtoken = StringToWString(token);
    for (int i = 0; i < wtoken.length(); i++)
    {
        std::wstring tmp = wtoken.substr(i, 1);
        idx.push_back(tokenizer_token2idx[tmp]);
    }
    return idx;
}

std::string idx2token(std::map<int, std::wstring>& tokenizer_idx2token, std::vector<int> idx)
{
    std::wstring wtoken;
    for (int i = 0; i < idx.size(); i++)
    {
        wtoken += tokenizer_idx2token[idx[i]];
    }
    std::string token = WStringToString(wtoken);
    return token;
}

#endif // BENCHMARK_UTILS_H