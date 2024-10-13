#include "flexnnschedule.h"

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <memory_profile_path> <time_profile_path> <malloc_plan_path> <layer_dependency_path> <memory_budget> [<skip count> <memory_layout_path>]\n", argv[0]);
        return -1;
    }

    double start = flexnn::get_current_time();

    const char* memory_profile_path = argv[1];
    const char* time_profile_path = argv[2];
    const char* malloc_plan_path = argv[3];
    const char* layer_dependency_path = argv[4];
    const int memory_budget = atoi(argv[5]);

    char memory_layout_path[256];
    memory_layout_path[0] = '\0';
    if (argc >= 8)
    {
        strcpy(memory_layout_path, argv[7]);
    }

    FlexnnSchedule scheduler;

    if (argc >= 7)
    {
        int skip_count = atoi(argv[6]);
        scheduler.m_skip_layer_count = skip_count;
    }

    scheduler.read_profiles(memory_profile_path, time_profile_path);

    scheduler.schedule_naive(memory_budget);

    if (strcmp(memory_layout_path, "") == 0)
    {
        scheduler.generate_write_schedule(malloc_plan_path, layer_dependency_path);
    }
    else
    {
        scheduler.generate_write_schedule(malloc_plan_path, layer_dependency_path, memory_layout_path);
    }
    scheduler.print_predicted_latency();

    double end = flexnn::get_current_time();
    fprintf(stderr, "total scheduling time: %.2f ms\n", end - start);

    return 0;
}