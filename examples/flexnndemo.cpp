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

#include "flexnnschedule.h"
#include "flexnnslice.h"

// #include std

// fixed model and config
// static const char* ncnnparam = "/data/local/tmp/models/ncnn/vgg19.ncnn.param";
// static const char* ncnnbin = "/data/local/tmp/models/ncnn/vgg19.ncnn.bin";
// static const char* flexnnparam = "/data/local/tmp/models/flexnn/vgg19.flexnn.param";
// static const char* flexnnbin = "/data/local/tmp/models/flexnn/vgg19.flexnn.bin";

static char ncnnparam[256];
static char ncnnbin[256];
static char flexnnparam[256];
static char flexnnbin[256];

static int g_warmup_loop_count = 0;
static int g_loop_count = 32;
static int g_cooling_down_duration = 0; // seconds
static bool g_enable_cooling_down = false;
static bool g_load_model_bin = true;
static int g_computing_powersave = 2;
static int g_loading_powersave = 3;
static int g_num_threads = 2;

static const char* input_shape = "[1,3,224,224]";

// for infer
static flexnn::PlannedAllocator g_planned_allocator;
static flexnn::PlannedAllocatorInterface g_planned_weight_allocator;
static flexnn::PlannedAllocatorInterface g_planned_blob_allocator;
static flexnn::PlannedAllocatorInterface g_planned_intermediate_allocator;
static flexnn::LockedTimeProfiler g_locked_time_profiler;

// interfaces
static std::vector<std::vector<int> > g_malloc_offsets;
static std::vector<int> g_persistent_offsets;
static std::vector<int> g_layer_dependencies;
static std::vector<MemoryProfilerEvent> g_memory_profiles;
static std::vector<LayerTimeProfile> g_time_profiles;

// for profiler
static flexnn::MemoryProfiler g_memory_profiler;
static flexnn::MemoryProfilerInterface g_weight_interface;
static flexnn::MemoryProfilerInterface g_blob_interface;
static flexnn::MemoryProfilerInterface g_intermediate_interface;
static flexnn::UnlockedTimeProfiler g_unlocked_time_profiler;

static std::vector<double> starts, ends;
// todo: profiles, plans, interfaces

int run_slice(int conv_mem, int fc_mem)
{
    double start = flexnn::get_current_time();

    int max_conv_size = conv_mem / 4;
    int max_fc_size = fc_mem / 4;
    FlexnnSlice slicer;

    slicer.storage_type = 0;

    // fprintf(stderr, "load model %s %s\n", inparam, inbin);

    slicer.load_param_dummy(ncnnparam);

    if (strcmp(ncnnbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        slicer.load_model(dr);
        slicer.gen_random_weight = true;
    }
    else
    {
        int ret = slicer.load_model(ncnnbin);
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

    slicer.save(flexnnparam, flexnnbin);

    double end = flexnn::get_current_time();
    // fprintf(stderr, "total slicing time %.2f ms\n", end - start);

    return 0;
}

int run_profile()
{
    double start = flexnn::get_current_time();

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, "flexnn_profile", g_num_threads);

    g_memory_profiler.clear();
    g_unlocked_time_profiler.clear();

    opt.blob_allocator = &g_blob_interface;
    opt.weight_allocator = &g_weight_interface;
    opt.workspace_allocator = &g_intermediate_interface;
    opt.time_profiler = &g_unlocked_time_profiler;

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    // profile
    ncnn::Mat in = cstr2mat(input_shape);
    in.fill(0.01f);

    ncnn::Net net;

    net.opt = opt;

    net.load_param(flexnnparam);

    if (g_load_model_bin)
    {
        net.load_model(flexnnbin);
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

    out.release();
    ex.clear();
    net.clear();

    g_memory_profiler.save(g_memory_profiles);
    g_unlocked_time_profiler.save(g_time_profiles);

    double end = flexnn::get_current_time();
    double time = end - start;
    // fprintf(stderr, "total profiling time: %.2f ms\n", time);

    return 0;
}

int run_schedule(int memory_budget)
{
    double start = flexnn::get_current_time();

    FlexnnSchedule scheduler;
    scheduler.set_memory_profiles(g_memory_profiles);
    scheduler.set_time_profiles(g_time_profiles);
    scheduler.schedule_naive(memory_budget);
    scheduler.get_malloc_plan(g_malloc_offsets, g_persistent_offsets);
    scheduler.get_layer_dependencies(g_layer_dependencies);
    double end = flexnn::get_current_time();
    // fprintf(stderr, "total scheduling time: %.2f ms\n", end - start);
    fprintf(stderr, "scheduling start: %.3f s, ends: %.3f s\n", start, end);

    return 0;
}

int run_infer(int memory_budget, int loop)
{
    double inf_start = flexnn::get_current_time();
    // fprintf(stderr, "run inference, memory budget: %d, loops: %d\n", memory_budget, loop);
    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, "flexnn_parallel", g_num_threads);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    // fprintf(stderr, "prepare allocator\n");
    opt.weight_allocator = &g_planned_weight_allocator;
    opt.blob_allocator = &g_planned_blob_allocator;
    opt.workspace_allocator = &g_planned_intermediate_allocator;
    g_planned_allocator.init_buffer(memory_budget);
    g_planned_allocator.set_malloc_plan(g_malloc_offsets, g_persistent_offsets);

    opt.layer_dependencies = &g_layer_dependencies;

    // flexnn::print_vector<int>(*opt.layer_dependencies);

    // fprintf(stderr, "start inference\n");

    ncnn::Mat in = cstr2mat(input_shape);
    in.fill(0.01f);

    g_planned_allocator.clear();

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    net.load_param(flexnnparam);

    // load persistent weights if have
    g_planned_allocator.set_load_mode(0);
    ncnn::Option opt2 = opt;
    opt2.use_parallel_preloading = false;
    opt2.use_ondemand_loading = false;
    net.opt = opt2;
    if (g_load_model_bin)
    {
        if (net.load_model(flexnnbin))
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
        if (net.load_model(flexnnbin))
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

    for (int i = 0; i < loop; i++)
    {
        double start = flexnn::get_current_time();

        starts.push_back(start);

        {
            g_planned_allocator.clear();
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out);
        }

        double end = flexnn::get_current_time();

        ends.push_back(end);

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;

        // fprintf(stderr, "loop %d\t%7.2f\n", i, time);
    }

    if (opt.use_parallel_preloading)
    {
        net.clear_local_threads();
    }

    out.release();
    net.clear();
    g_planned_allocator.release_buffer();

    time_avg /= loop;

    // fprintf(stderr, "min = %7.2f  max = %7.2f  avg = %7.2f  load = %7.2f\n", time_min, time_max, time_avg, load_end - load_start);
    double inf_end = flexnn::get_current_time();
    fprintf(stderr, "inference start: %.3f s, ends: %.3f s\n", inf_start, inf_end);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <ncnn_param> <ncnn_bin> <flexnn_param> <flexnn_bin> <result_path> [<idle_duration>]\n", argv[0]);
        return -1;
    }

    strcpy(ncnnparam, argv[1]);
    strcpy(ncnnbin, argv[2]);
    strcpy(flexnnparam, argv[3]);
    strcpy(flexnnbin, argv[4]);

    int num_configs = 4;
    int loops[4] = {32, 32, 32, 32};
    const int m = 1e6;
    int memory_budgets[4] = {100 * m, 300 * m, 200 * m, 500 * m};
    int conv_sz[4] = {100 * m, 0, 0, 0};
    int fc_sz[4] = {20 * m, 0, 0, 0};
    int idle_duration = 0;

    if (argc > 6)
    {
        idle_duration = atoi(argv[6]);
    }

    // init

    g_weight_interface.set_attributes(0, 0);
    g_blob_interface.set_attributes(0, 1);
    g_intermediate_interface.set_attributes(0, 2);

    g_memory_profiler.add(&g_weight_interface);
    g_memory_profiler.add(&g_blob_interface);
    g_memory_profiler.add(&g_intermediate_interface);

    g_planned_weight_allocator.set_attributes(0);
    g_planned_blob_allocator.set_attributes(1);
    g_planned_intermediate_allocator.set_attributes(2);
    g_planned_allocator.add(&g_planned_weight_allocator);
    g_planned_allocator.add(&g_planned_blob_allocator);
    g_planned_allocator.add(&g_planned_intermediate_allocator);

    cooling_down(idle_duration > 0, idle_duration); // idle memory measure
    run_slice(conv_sz[0], fc_sz[0]);
    for (int i = 0; i < num_configs; i++)
    {
        run_profile();
        run_schedule(memory_budgets[i]);
        run_infer(memory_budgets[i], loops[i]);
    }

    // write starts and ends to file
    FILE* fp = fopen(argv[5], "w");
    fprintf(fp, "start(ms),end(ms)\n");
    for (int i = 0; i < starts.size(); i++)
    {
        fprintf(fp, "%.3f,%.3f\n", starts[i], ends[i]);
    }
    fclose(fp);

    return 0;
}