#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <stack>

// ncnn public header
#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"
#include "profiler.h"

// ncnn private header
#include "modelwriter.h"

// flexnn utils
#include "flexnn_utils.h"
#include "dummymat.h"

#include "benchmark_utils.h"
#include "profiler.h"
#include <fstream>

// global variables
static bool g_load_model_bin = true;

static flexnn::MemoryProfiler g_memory_profiler;
static flexnn::MemoryProfilerInterface g_weight_interface;
static flexnn::MemoryProfilerInterface g_blob_interface;
static flexnn::MemoryProfilerInterface g_intermediate_interface;
static flexnn::UnlockedTimeProfiler g_time_profiler;

void profile_gpt2(const char* comment, const char* vocabpath, const ncnn::Option& opt)
{
    const int max_history_len = 3;
    const int max_len = 1;

    std::map<std::wstring, int> tokenizer_token2idx;
    std::map<int, std::wstring> tokenizer_idx2token;
    std::vector<std::vector<int> > history;

    // load vocab
    {
        std::ifstream infile;
        infile.open(vocabpath);
        std::string line;
        int idx = 0;
        while (getline(infile, line))
        {
            auto ws = StringToWString(line);
            tokenizer_token2idx.insert(std::pair<std::wstring, int>(ws, idx));
            tokenizer_idx2token.insert(std::pair<int, std::wstring>(idx, ws));
            idx++;
        }
        infile.close();
    }

    std::string in = "你好你好你好你好你好";

    std::vector<int> text_ids = token2idx(tokenizer_token2idx, in);
    history.push_back(text_ids);
    std::vector<int> input_ids = {101};
    int history_len = 3;
    if (history.size() < max_history_len)
        history_len = history.size();
    std::vector<std::vector<int> > max_history;
    max_history.assign(history.end() - history_len, history.end());
    for (std::vector<int> history_utr : max_history)
    {
        input_ids = vector_merge(input_ids, history_utr);
        input_ids.push_back(102);
    }

    // prepare net

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    // g_load_model_bin = false;
    if (g_load_model_bin)
    {
        char binpath[256];
        sprintf(binpath, "%s.bin", comment);
        if (net.load_model(binpath))
        {
            // fall back to empty
            DataReaderFromEmpty dr; // load from empty
            net.load_model(dr);
        }
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    double load_end = flexnn::get_current_time();
    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    // infer
    std::vector<int> response;
    int it = 0;
    for (; it < max_len; it++)
    {
        fprintf(stderr, "Iteration %d\n", it);

        // print input
        fprintf(stderr, "input_ids: ");
        for (int i = 0; i < input_ids.size(); i++)
        {
            fprintf(stderr, "%d ", input_ids[i]);
        }
        fprintf(stderr, "\n");

        double start = flexnn::get_current_time();

        ncnn::Mat input_ids_mat(input_ids.size());
        ncnn::Mat position_ids_mat(input_ids.size());
        for (int i = 0; i < input_ids.size(); i++)
        {
            input_ids_mat[i] = float(input_ids[i]);
            position_ids_mat[i] = float(i);
        }

        const std::vector<const char*>& input_names = net.input_names();
        const std::vector<const char*>& output_names = net.output_names();

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], input_ids_mat);
            ex.input(input_names[1], position_ids_mat);
            ex.extract(output_names[0], logits);
            // ex.input("0", input_ids_mat);
            // ex.input("input.3", position_ids_mat);
            // ex.extract("1673", logits);
        }

        fprintf(stderr, "logits shape: [%d,%d]\n", logits.h, logits.w);

        ncnn::Mat next_token_logits;
        next_token_logits.clone_from(logits.row_range(logits.h - 1, 1));
        next_token_logits[100] = Neg_Infinity;
        top_k_filtering(next_token_logits);
        softmax<float>(next_token_logits, next_token_logits, 13317);
        int next_token = multinomial(next_token_logits);
        if (next_token == 102) break;
        response.push_back(next_token);
        input_ids.push_back(next_token);

        double end = flexnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;

        fprintf(stderr, "%20s  loop %d\t%7.2f\n", comment, it, time);
    }

    time_avg /= it;

    // history.push_back(response);
    std::string bot_text = idx2token(tokenizer_idx2token, response);

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f  load = %7.2f\n", comment, time_min, time_max, time_avg, load_end - load_start);
}

void profile(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    ncnn::Net net;

    net.opt = opt;

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    if (g_load_model_bin)
    {
        char binpath[256];
        sprintf(binpath, "%s.bin", comment);
        net.load_model(binpath);
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Mat out;

    double start = flexnn::get_current_time();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out);

    double end = flexnn::get_current_time();

    double time = end - start;
}

int main(int argc, char** argv)
{
    int num_threads = 1;
    // int num_threads = ncnn::get_physical_big_cpu_count();
    char input_shape[64];
    sprintf(input_shape, "[1,3,224,224]");
    char vocabpath[256];
    vocabpath[0] = '\0';

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <model_prefix> [<key=value>...]\n", argv[0]);
        fprintf(stderr, "  model_prefix: the model path w/o .param or .bin postfix\n");
        fprintf(stderr, "  memory_profile_path=model_prefix.memprof\n");
        fprintf(stderr, "  time_profile_path=model_prefix.timeprof\n");
        fprintf(stderr, "  num_threads=%d\n", num_threads);
        fprintf(stderr, "  inputshape=%s\n", input_shape);
        fprintf(stderr, "  vocab_path=%s\n", vocabpath);
        fprintf(stderr, "Example: %s ~/models/flexnn/vgg19.flexnn loop_count=4 warmup_loop_count=0\n", argv[0]);
        return -1;
    }

    double start = flexnn::get_current_time();

    const char* model_prefix = argv[1];
    char memory_profile_path[256];
    strcpy(memory_profile_path, model_prefix);
    strcat(memory_profile_path, ".memprof");
    char time_profile_path[256];
    strcpy(time_profile_path, model_prefix);
    strcat(time_profile_path, ".timeprof");

    for (int i = 2; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;
        if (strcmp(key, "num_threads") == 0)
            num_threads = atoi(value);
        if (strcmp(key, "input_shape") == 0)
            strcpy(input_shape, value);
        if (strcmp(key, "memory_profile_path") == 0)
            strcpy(memory_profile_path, value);
        if (strcmp(key, "time_profile_path") == 0)
            strcpy(time_profile_path, value);
        if (strcmp(key, "vocab_path") == 0)
            strcpy(vocabpath, value);
    }

    // g_blob_pool_allocator.set_size_compare_ratio(0.f);
    // g_workspace_pool_allocator.set_size_compare_ratio(0.f);

    g_weight_interface.set_attributes(0, 0);
    g_blob_interface.set_attributes(0, 1);
    g_intermediate_interface.set_attributes(0, 2);

    g_memory_profiler.add(&g_weight_interface);
    g_memory_profiler.add(&g_blob_interface);
    g_memory_profiler.add(&g_intermediate_interface);

    // print
    fprintf(stderr, "  model_prefix=%s\n", model_prefix);
    fprintf(stderr, "  memory_profile_path=%s\n", memory_profile_path);
    fprintf(stderr, "  time_profile_path=%s\n", time_profile_path);
    fprintf(stderr, "  num_threads=%d\n", num_threads);
    fprintf(stderr, "  inputshape=%s\n", input_shape);
    if (strcmp(vocabpath, "") != 0)
        fprintf(stderr, "  vocab_path=%s\n", vocabpath);

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, "flexnn_profile", num_threads);

    opt.blob_allocator = &g_blob_interface;
    opt.weight_allocator = &g_weight_interface;
    opt.workspace_allocator = &g_intermediate_interface;
    opt.time_profiler = &g_time_profiler;

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);
    ncnn::set_cpu_powersave(0); // omp use big cores only

    // gpt2
    if (strstr(model_prefix, "gpt2"))
    {
        profile_gpt2(model_prefix, vocabpath, opt);
        g_memory_profiler.save(memory_profile_path);
        g_time_profiler.save(time_profile_path);

        double end = flexnn::get_current_time();
        double time = end - start;
        fprintf(stderr, "total profiling time: %.2f ms\n", time);
        return 0;
    }

    // profile
    ncnn::Mat in = cstr2mat(input_shape);
    profile(model_prefix, in, opt);

    g_memory_profiler.save(memory_profile_path);
    g_time_profiler.save(time_profile_path);

    double end = flexnn::get_current_time();
    double time = end - start;
    fprintf(stderr, "total profiling time: %.2f ms\n", time);

    return 0;
}
