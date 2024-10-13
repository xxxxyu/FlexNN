#include "benchmark_utils.h"
#include "plannedallocator.h"
#include "profiler.h"
#include <fstream>
#include "flexnn_utils.h"

// global variables
static int g_warmup_loop_count = 4;
static int g_loop_count = 8;
static int g_cooling_down_duration = 0; // seconds
static bool g_enable_cooling_down = false;
static bool g_load_model_bin = true;
static int g_computing_powersave = 2;
static int g_loading_powersave = 3;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
static flexnn::PlannedAllocator g_planned_allocator;
static flexnn::PlannedAllocatorInterface g_planned_weight_allocator;
static flexnn::PlannedAllocatorInterface g_planned_blob_allocator;
static flexnn::PlannedAllocatorInterface g_planned_intermediate_allocator;
static flexnn::LockedTimeProfiler g_time_profiler;

void benchmark_gpt2(const char* comment, const char* vocabpath, const ncnn::Option& opt)
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

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    g_planned_allocator.clear();

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    // load persistent weights if have
    g_planned_allocator.set_load_mode(0);
    ncnn::Option opt2 = opt;
    opt2.use_parallel_preloading = false;
    opt2.use_ondemand_loading = false;
    net.opt = opt2;
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
    // reset opt
    net.opt = opt;
    // reset mode
    g_planned_allocator.set_load_mode(1);

    g_planned_allocator.clear();

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

    if (opt.use_parallel_preloading && opt.use_local_threads)
    {
        fprintf(stderr, "use_parallel_preloading and use_local_threads.\n");
        net.initialize_local_threads(g_computing_powersave, g_loading_powersave);
    }

    double load_end = flexnn::get_current_time();
    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int j = 0; j < g_loop_count + g_warmup_loop_count; j++)
    {
        g_planned_allocator.clear();
        // infer
        std::vector<int> response;
        int it = 0;

        double start = flexnn::get_current_time();
        for (; it < max_len; it++)
        {
            // fprintf(stderr, "Iteration %d\n", it);

            // print input
            fprintf(stderr, "input_ids: ");
            for (int i = 0; i < input_ids.size(); i++)
            {
                fprintf(stderr, "%d ", input_ids[i]);
            }
            fprintf(stderr, "\n");

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
                // // print input shapes
                // fprintf(stderr, "input_ids shape: [%d,%d]\n", input_ids_mat.h, input_ids_mat.w);
                // fprintf(stderr, "position_ids shape: [%d,%d]\n", position_ids_mat.h, position_ids_mat.w);
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
            // response.push_back(next_token);
            // input_ids.push_back(next_token);
        }
        double end = flexnn::get_current_time();

        double time = end - start;

        if (j >= g_warmup_loop_count)
        {
            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;

            fprintf(stderr, "%20s  loop %d\t%7.2f ms\n", comment, j - g_warmup_loop_count, time);
        }

        // history.push_back(response);
        std::string bot_text = idx2token(tokenizer_idx2token, response);
    }

    time_avg /= g_loop_count;

    if (opt.use_parallel_preloading)
    {
        net.clear_local_threads();
    }

    fprintf(stderr, "%20s  min = %7.2f ms  max = %7.2f ms  avg = %7.2f ms  load = %7.2f ms\n", comment, time_min, time_max, time_avg, load_end - load_start);
}

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    // fprintf(stderr, "Benchmark input shape: [%d,%d,%d,%d]\n", in.d, in.c, in.h, in.w);
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    g_planned_allocator.clear();

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    // load persistent weights if have
    g_planned_allocator.set_load_mode(0);
    ncnn::Option opt2 = opt;
    opt2.use_parallel_preloading = false;
    opt2.use_ondemand_loading = false;
    net.opt = opt2;
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
    // reset opt
    net.opt = opt;
    // reset mode
    g_planned_allocator.set_load_mode(1);

    g_planned_allocator.clear();

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

    if (opt.use_parallel_preloading && opt.use_local_threads)
    {
        fprintf(stderr, "use_parallel_preloading and use_local_threads.\n");
        net.initialize_local_threads(g_computing_powersave, g_loading_powersave);
    }

    double load_end = flexnn::get_current_time();

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    cooling_down(g_enable_cooling_down, g_cooling_down_duration); // cooling down if enabled

    ncnn::Mat out;

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        g_planned_allocator.clear();
        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = flexnn::get_current_time();

        {
            g_planned_allocator.clear();
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
        }

        double end = flexnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;

        fprintf(stderr, "%20s  loop %d\t%7.2f ms\n", comment, i, time);
    }

    if (opt.use_parallel_preloading)
    {
        net.clear_local_threads();
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f ms  max = %7.2f ms  avg = %7.2f ms  load = %7.2f ms\n", comment, time_min, time_max, time_avg, load_end - load_start);
}

void benchmark_compare(const char* comment_a, const char* comment_b, const ncnn::Mat& _in, const ncnn::Option& opt) // compare results of 2 models, used for correctness check
{
    ncnn::Mat in = _in;
    fprintf(stderr, "Benchmark input shape: [%d,%d,%d,%d]\n", in.d, in.c, in.h, in.w);
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net nets[2];
    const char* comments[2] = {comment_a, comment_b};
    ncnn::Mat outs[2];

    // inference
    for (int i = 0; i < 2; i++)
    {
        ncnn::Net& net = nets[i];
        const char* comment = comments[i];

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

        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out);

        outs[i] = out.clone();
    }

    // comparation
    if (matcmp(outs[0], outs[1]))
    {
        fprintf(stderr, "Correctness check failed.\n");
    }
    else
    {
        fprintf(stderr, "Correctness check passed.\n");
    }
}

int main(int argc, char** argv)
{
    int loop_count = g_loop_count;
    int num_threads = 1;
    // int num_threads = ncnn::get_physical_big_cpu_count();
    int cooling_down_duration = g_cooling_down_duration;
    int warmup_loop_count = g_warmup_loop_count;
    char input_shape[64];
    sprintf(input_shape, "[1,3,224,224]");
    char cmp_model_prefix[256];
    cmp_model_prefix[0] = '\0';
    char config[64];
    sprintf(config, "ncnn_parallel");
    char malloc_plan_path[256];
    malloc_plan_path[0] = '\0';
    char layer_dependency_path[256];
    layer_dependency_path[0] = '\0';
    char time_profile_path[256];
    time_profile_path[0] = '\0';
    char vocabpath[256];
    vocabpath[0] = '\0';
    int memory_budget = -1;
    int computing_powersave = -1;
    int loading_powersave = -1;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <model_prefix> [<key=value>...]\n", argv[0]);
        fprintf(stderr, "  model_prefix: the model path w/o .param or .bin postfix\n");
        fprintf(stderr, "  loop_count=%d\n", g_loop_count);
        fprintf(stderr, "  warmup_loop_count=%d\n", g_warmup_loop_count);
        fprintf(stderr, "  cooling_down_duration=%d (s)\n", g_cooling_down_duration);
        fprintf(stderr, "  num_threads=%d\n", num_threads);
        fprintf(stderr, "  computing_powersave=%d\n", g_computing_powersave);
        fprintf(stderr, "  loading_powersave=%d\n", g_loading_powersave);
        fprintf(stderr, "  input_shape=%s\n", input_shape);
        fprintf(stderr, "  cmp_model_prefix=%s\n", cmp_model_prefix);
        fprintf(stderr, "  config=%s\n", config);
        fprintf(stderr, "  malloc_plan_path=%s\n", malloc_plan_path);
        fprintf(stderr, "  layer_dependency_path=%s\n", layer_dependency_path);
        fprintf(stderr, "  time_profile_path=%s\n", time_profile_path);
        fprintf(stderr, "  memory_budget=%d\n", memory_budget);
        fprintf(stderr, "  vocab_path=%s\n", vocabpath);
        fprintf(stderr, "Example: %s ~/models/flexnn/vgg19.flexnn loop_count=4 warmup_loop_count=0\n", argv[0]);
        return -1;
    }

    const char* model_prefix = argv[1];

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

        if (strcmp(key, "loop_count") == 0)
            loop_count = atoi(value);
        if (strcmp(key, "warmup_loop_count") == 0)
            warmup_loop_count = atoi(value);
        if (strcmp(key, "cooling_down_duration") == 0)
            cooling_down_duration = atoi(value);
        if (strcmp(key, "num_threads") == 0)
            num_threads = atoi(value);
        if (strcmp(key, "input_shape") == 0)
            strcpy(input_shape, value);
        if (strcmp(key, "cmp_model_prefix") == 0)
            strcpy(cmp_model_prefix, value);
        if (strcmp(key, "config") == 0)
            strcpy(config, value);
        if (strcmp(key, "malloc_plan_path") == 0)
            strcpy(malloc_plan_path, value);
        if (strcmp(key, "layer_dependency_path") == 0)
            strcpy(layer_dependency_path, value);
        if (strcmp(key, "time_profile_path") == 0)
            strcpy(time_profile_path, value);
        if (strcmp(key, "memory_budget") == 0)
            memory_budget = atoi(value);
        if (strcmp(key, "computing_powersave") == 0)
            computing_powersave = atoi(value);
        if (strcmp(key, "loading_powersave") == 0)
            loading_powersave = atoi(value);
        if (strcmp(key, "vocab_path") == 0)
            strcpy(vocabpath, value);
    }

    // set global variables
    g_loop_count = loop_count;
    g_warmup_loop_count = warmup_loop_count;
    g_enable_cooling_down = cooling_down_duration > 0;
    g_cooling_down_duration = cooling_down_duration;

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

    g_computing_powersave = computing_powersave >= 0 ? computing_powersave : g_computing_powersave;
    g_loading_powersave = loading_powersave >= 0 ? loading_powersave : g_loading_powersave;

    // print
    fprintf(stderr, "  model_prefix=%s\n", model_prefix);
    fprintf(stderr, "  loop_count=%d\n", g_loop_count);
    fprintf(stderr, "  warmup_loop_count=%d\n", g_warmup_loop_count);
    fprintf(stderr, "  cooling_down_duration=%ds\n", g_cooling_down_duration);
    fprintf(stderr, "  num_threads=%d\n", num_threads);
    fprintf(stderr, "  computing_powersave=%d\n", g_computing_powersave);
    fprintf(stderr, "  loading_powersave=%d\n", g_loading_powersave);
    fprintf(stderr, "  inputshape=%s\n", input_shape);
    if (strcmp(cmp_model_prefix, "") != 0)
        fprintf(stderr, "  cmp_model_prefix=%s\n", cmp_model_prefix);
    fprintf(stderr, "  config=%s\n", config);
    if (strcmp(malloc_plan_path, "") != 0)
        fprintf(stderr, "  malloc_plan_path=%s\n", malloc_plan_path);
    if (strcmp(layer_dependency_path, "") != 0)
        fprintf(stderr, "  layer_dependency_path=%s\n", layer_dependency_path);
    if (strcmp(time_profile_path, "") != 0)
        fprintf(stderr, "  time_profile_path=%s\n", time_profile_path);
    if (memory_budget > 0)
        fprintf(stderr, "  memory_budget=%d\n", memory_budget);
    if (strcmp(vocabpath, "") != 0)
        fprintf(stderr, "  vocab_path=%s\n", vocabpath);

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, config, num_threads);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    if (strcmp(malloc_plan_path, "") != 0)
    {
        opt.weight_allocator = &g_planned_weight_allocator;
        opt.blob_allocator = &g_planned_blob_allocator;
        opt.workspace_allocator = &g_planned_intermediate_allocator;
        g_planned_weight_allocator.set_attributes(0);
        g_planned_blob_allocator.set_attributes(1);
        g_planned_intermediate_allocator.set_attributes(2);
        g_planned_allocator.add(&g_planned_weight_allocator);
        g_planned_allocator.add(&g_planned_blob_allocator);
        g_planned_allocator.add(&g_planned_intermediate_allocator);
        g_planned_allocator.init_buffer(memory_budget);
        g_planned_allocator.load_malloc_plan(malloc_plan_path);
    }

    std::vector<int> layer_dependencies;
    if (strcmp(layer_dependency_path, "") != 0)
    {
        load_layer_dependency(layer_dependency_path, layer_dependencies);
        opt.layer_dependencies = &layer_dependencies;
    }

    if (strcmp(time_profile_path, "") != 0)
    {
        opt.time_profiler = &g_time_profiler;
    }

    if (g_enable_cooling_down)
    {
#ifdef _WIN32
        Sleep(g_cooling_down_duration * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(g_cooling_down_duration);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = g_cooling_down_duration;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    // gpt2
    if (strstr(model_prefix, "gpt2"))
    {
        benchmark_gpt2(model_prefix, vocabpath, opt);
        if (strcmp(time_profile_path, "") != 0)
        {
            g_time_profiler.save(time_profile_path);
        }
        return 0;
    }

    // benchmark
    ncnn::Mat in = cstr2mat(input_shape);
    if (strcmp(cmp_model_prefix, "") == 0)
        benchmark(model_prefix, in, opt);
    else
        benchmark_compare(model_prefix, cmp_model_prefix, in, opt);

    if (strcmp(time_profile_path, "") != 0)
    {
        g_time_profiler.save(time_profile_path);
    }

    return 0;
}
