#ifndef FLEXNN_SCHEDULE_H
#define FLEXNN_SCHEDULE_H
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "profiler.h"

#include <vector>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <iostream>
#include <list>
#include <algorithm>
#include <map>
#include <cmath>
#include <iterator>

#include "allocator.h"
#include "xyplane.h"

class MemoryProfile
{
public:
    MemoryProfile()
        : start_layer_index(0), end_layer_index(0), size(0), memory_type(0), malloc_count(0), x(0), y(0)
    {
    }

public:
    int start_layer_index; // start of lifetime
    int end_layer_index;   // end of lifetime
    int size;

    int memory_type;  // 0 for weight, 1 for blob, 2 for intermediate
    int malloc_count; // malloc count of this type

    // schedule
    int x; // malloc time
    int y; // offset

public:
    int memory_index() const // get a uuid based on mem_t and m_cnt
    {
        return ((x & 0x3fff) << 18) | ((memory_type & 0x3) << 16) | (malloc_count & 0xffff); // 14 bits for x, 2 bits for mem_t, 16 bits for m_cnt
    }

    static int memory_index(int x, int mem_t, int m_cnt) // get a uuid based on mem_t and m_cnt
    {
        return ((x & 0x3fff) << 18) | ((mem_t & 0x3) << 30) | (m_cnt & 0xffff); // 14 bits for x, 2 bits for mem_t, 16 bits for m_cnt
    }

public:
    // compare
    static bool compare_layer_index(const MemoryProfile& a, const MemoryProfile& b)
    {
        return a.start_layer_index < b.start_layer_index;
    }
    static bool compare_size(const MemoryProfile& a, const MemoryProfile& b)
    {
        return a.size < b.size;
    }
};

using flexnn::LayerTimeProfile;
using flexnn::MemoryProfilerEvent;

// return >= offset, aligned to NCNN_MALLOC_ALIGN
static inline size_t alignOffsetBig(size_t offset)
{
    return ncnn::alignSize(offset, NCNN_MALLOC_ALIGN);
}

// return <= offset, aligned to NCNN_MALLOC_ALIGN
static inline size_t alignOffsetSmall(size_t offset)
{
    return ncnn::alignSize(offset - NCNN_MALLOC_ALIGN + 1, NCNN_MALLOC_ALIGN);
}

class FlexnnSchedule
{
public:
    FlexnnSchedule()
    {
        m_xy_plane = 0;
    }
    ~FlexnnSchedule()
    {
        if (m_xy_plane)
            delete m_xy_plane;
    }

    int init_xyplane(int x, int y)
    {
        if (m_xy_plane)
            delete m_xy_plane;
        m_xy_plane = new XYPlane(x, y, NCNN_MALLOC_ALIGN);
        return 0;
    }
    int read_profiles(const char* memory_profile_path, const char* time_profile_path);
    int read_memory_profile(const char* path);
    int read_time_profile(const char* path);
    int memory_events_to_profiles();

    // schedule functions: inputs -> memory_schedule
    int schedule_naive(const int memory_budget);

    // memory_schedule -> layer_dependency
    int resolve_layer_dependencies(const std::map<int, MemoryProfile>& memory_schedule, std::vector<int>& layer_dependencies);

    // predictor: layer_denpendencies -> latency
    double predict_latency(const std::vector<int>& layer_dependencies);

    // memory_schedule -> malloc_plan
    int generate_malloc_plan(const std::map<int, MemoryProfile>& memory_schedule, std::vector<std::vector<int> >& malloc_plan);

    // write to file
    int write_malloc_plan(const char* path) const;
    int write_layer_dependencies(const char* path) const;
    int write_memory_layout(const char* path) const;

    int generate_write_schedule(const char* malloc_plan_path, const char* layer_dependency_path, const char* memory_layout_path = 0);
    void print_predicted_latency();

    int get_layer_count() const;
    double get_total_loading_duration() const;
    double get_total_computing_duration() const;
    // int get_

    int get_malloc_plan(std::vector<std::vector<int> >& malloc_offsets, std::vector<int>& persistent_offsets)
    {
        generate_malloc_plan(m_memory_schedule, m_malloc_plan);
        malloc_offsets = m_malloc_plan;
        persistent_offsets = m_persistent_offsets;
        return 0;
    }
    int get_layer_dependencies(std::vector<int>& layer_dependencies)
    {
        resolve_layer_dependencies(m_memory_schedule, m_layer_dependencies);
        layer_dependencies = m_layer_dependencies;
        return 0;
    }
    int set_memory_profiles(const std::vector<MemoryProfilerEvent>& memory_profiler_events)
    {
        m_memory_profiler_events = memory_profiler_events;
        fprintf(stderr, "read %d memory events\n", (int)m_memory_profiler_events.size());
        return memory_events_to_profiles();
    }
    int set_time_profiles(const std::vector<LayerTimeProfile>& time_profiles)
    {
        m_time_profiles = time_profiles;
        fprintf(stderr, "read %d time profiles\n", (int)m_time_profiles.size());
        return 0;
    }

public:
    // outputs
    std::vector<int> m_layer_dependencies;
    std::vector<std::vector<int> > m_malloc_plan;

    // temp
    std::map<int, MemoryProfile> m_memory_schedule; // b.first=x=time, b.second=y=memory, sorted by x, for same x sorted by type and count (index)
    std::vector<int> m_persistent_offsets;
    std::vector<double> m_loading_begin;
    std::vector<double> m_loading_end;
    std::vector<double> m_computing_begin;
    std::vector<double> m_computing_end;

    // inputs
    std::vector<MemoryProfilerEvent> m_memory_profiler_events;
    std::vector<LayerTimeProfile> m_time_profiles;
    std::map<int, MemoryProfile> m_memory_profiles; // x=0, sorted by m_type and m_cnt
    int m_weight_count;
    int m_blob_count;
    int m_intermediate_count;

    // const
    int m_skip_layer_count = 1;

    XYPlane* m_xy_plane;
};

int FlexnnSchedule::get_layer_count() const
{
    int max_index = 0;
    for (auto profile : m_time_profiles)
    {
        if (profile.layer_index > max_index)
        {
            max_index = profile.layer_index;
        }
    }
    return max_index + 1;
}

int FlexnnSchedule::read_profiles(const char* memory_profile_path, const char* time_profile_path)
{
    read_memory_profile(memory_profile_path);
    read_time_profile(time_profile_path);

    return 0;
}

void FlexnnSchedule::print_predicted_latency()
{
    fprintf(stderr, "predicted latency: %f\n", predict_latency(m_layer_dependencies));
}

double FlexnnSchedule::get_total_loading_duration() const
{
    double total = 0;
    for (auto profile : m_time_profiles)
    {
        total += profile.loading_duration;
    }
    return total;
}

double FlexnnSchedule::get_total_computing_duration() const
{
    double total = 0;
    for (auto profile : m_time_profiles)
    {
        total += profile.computing_duration;
    }
    return total;
}

int FlexnnSchedule::read_memory_profile(const char* path)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    char line[256];

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line[0] == '#') // comment
            continue;
        if (strcmp(line, "layer_index,memory_type,event_type,ptr,size,time\n") == 0) // first line
            continue;

        flexnn::MemoryProfilerEvent event;
        int ret = sscanf(line, "%d,%d,%d,%p,%zu,%lf\n", &event.layer_index, &event.memory_type, &event.event_type, &event.ptr, &event.size, &event.time);
        if (ret != 6)
        {
            fprintf(stderr, "fscanf failed\n");
            break;
        }

        m_memory_profiler_events.push_back(event);
    }

    fprintf(stderr, "read %d memory events\n", (int)m_memory_profiler_events.size());

    return memory_events_to_profiles();
}

int FlexnnSchedule::memory_events_to_profiles()
{
    // events are ordered by malloc - free

    std::map<void*, int> memory_indices;
    std::map<int, bool> profile_paired;
    int counters[3] = {0, 0, 0}; // malloc count of each type, 0 for weight, 1 for blob, 2 for intermediate
    int malloc_count = 0, free_count = 0;

    for (int i = 0; i < (int)m_memory_profiler_events.size(); i++)
    {
        const flexnn::MemoryProfilerEvent& event = m_memory_profiler_events[i];

        if (event.event_type == 1)
        {
            MemoryProfile profile;
            profile.start_layer_index = event.layer_index;
            profile.size = event.size;
            profile.memory_type = event.memory_type;
            profile.malloc_count = counters[event.memory_type]++;
            memory_indices[event.ptr] = profile.memory_index();
            m_memory_profiles.insert({profile.memory_index(), profile});
            profile_paired.insert({profile.memory_index(), false});
            malloc_count++;
        }
        else if (event.event_type == 0)
        {
            if (memory_indices.find(event.ptr) == memory_indices.end())
            {
                fprintf(stderr, "free event with no pair malloc to pair %d %p\n", event.layer_index, event.ptr);
                // return -1;
                continue;
            }
            if (profile_paired[memory_indices[event.ptr]])
            {
                fprintf(stderr, "free event with already paired malloc %d %p\n", event.layer_index, event.ptr);
                // return -1;
                continue;
            }
            m_memory_profiles[memory_indices[event.ptr]].end_layer_index = event.layer_index;
            profile_paired[memory_indices[event.ptr]] = true;
            memory_indices.erase(event.ptr);
            free_count++;
        }
    }

    if (!memory_indices.empty())
    {
        fprintf(stderr, "memory free not detected:\n");
        for (auto it = memory_indices.begin(); it != memory_indices.end(); it++)
        {
            fprintf(stderr, "%p %d\n", it->first, it->second);
            fprintf(stderr, "%p allocated at layer %d\n", it->first, m_memory_profiles[it->second].start_layer_index);
        }
        return -1;
    }

    m_weight_count = counters[0];
    m_blob_count = counters[1];
    m_intermediate_count = counters[2];

    fprintf(stderr, "get %d memory profiles, malloc_count=%d, free_count=%d\n", (int)m_memory_profiles.size(), malloc_count, free_count);

    return 0;
}

int FlexnnSchedule::read_time_profile(const char* path)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    char line[256];

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line[0] == '#') // comment
            continue;
        if (strcmp(line, "layer_index,loading_begin,loading_end,loading_duration,computing_begin,computing_end,computing_duration\n") == 0) // first line
            continue;

        flexnn::LayerTimeProfile profile;
        int ret = sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", &profile.layer_index, &profile.loading_begin, &profile.loading_end, &profile.loading_duration, &profile.computing_begin, &profile.computing_end, &profile.computing_duration);
        if (ret != 7)
        {
            fprintf(stderr, "fscanf failed\n");
            break;
        }

        m_time_profiles.push_back(profile);
    }

    fprintf(stderr, "read %d time profiles\n", (int)m_time_profiles.size());

    // print total loading and computing
    fprintf(stderr, "total loading: %f, total computing: %f\n", get_total_loading_duration(), get_total_computing_duration());

    return 0;
}

int FlexnnSchedule::generate_malloc_plan(const std::map<int, MemoryProfile>& memory_schedule, std::vector<std::vector<int> >& malloc_plan)
{
    malloc_plan.resize(3);

    // schedule is sorted by x, type and count
    for (auto schedule : memory_schedule)
    {
        if (schedule.second.y < 0)
        {
            fprintf(stderr, "invalid y: %d at %d of lid %d\n", schedule.second.y, schedule.second.x, schedule.second.start_layer_index);
            fprintf(stderr, "%d,%d,%d,%d,%d,%d,%d\n", schedule.second.start_layer_index, schedule.second.end_layer_index, schedule.second.size, schedule.second.memory_type, schedule.second.malloc_count, schedule.second.x, schedule.second.y);
            return -1;
        }
        malloc_plan[schedule.second.memory_type].push_back(schedule.second.y);
    }

    return 0;
}

int FlexnnSchedule::resolve_layer_dependencies(const std::map<int, MemoryProfile>& memory_schedule, std::vector<int>& layer_dependencies)
{
    // x_i < x_j && y_i '==' y_j

    // resolve computing - loading dependencies between layers
    // memory_schedule's key is x, value is a MemoryProfile
    // iterator all combination (i, j) of memory_schedule, with x_i<=x_j
    // if y_i==y_j, then i->j

    auto i = memory_schedule.begin();
    auto j = memory_schedule.begin();

    std::vector<int> last_layer_before_loading(get_layer_count(), -1);
    layer_dependencies.resize(get_layer_count(), get_layer_count());
    for (int i = 0; i < m_skip_layer_count; i++)
    {
        layer_dependencies[i] = m_skip_layer_count + 1;
    }

    // for (auto i = memory_schedule.begin(); i != memory_schedule.end(); i++)
    // {
    //     for (auto j = i; j != memory_schedule.end(); j++)
    //     {
    //         // skip j=i
    //         if (i == j)
    //             continue;

    //         // skip x_i==x_j
    //         if (i->second.x == j->second.x)
    //             continue;

    //         // skip if i is weight or j is not weight
    //         if (i->second.memory_type == 0 || j->second.memory_type != 0)
    //             continue;

    //         // if overlap, then has dependency
    //         if (i->second.y + i->second.size > j->second.y && j->second.y + j->second.size > i->second.y)
    //         {
    //             if (i->second.end_layer_index == j->second.start_layer_index)
    //             {
    //                 fprintf(stderr, "layer %d depends on next layer %d\n", i->second.end_layer_index - 1, j->second.start_layer_index);
    //                 fprintf(stderr, "i: %d %d %d %d\n", i->second.start_layer_index, i->second.end_layer_index, i->second.y, i->second.size);
    //                 fprintf(stderr, "j: %d %d %d %d\n", j->second.start_layer_index, j->second.end_layer_index, j->second.y, j->second.size);
    //                 return -1;
    //             }
    //             // i->j
    //             // layer_dependencies[i->second.end_layer_index - 1] = std::min(layer_dependencies[i->second.end_layer_index - 1], j->second.x);
    //             // layer_dependencies[i->second.end_layer_index] = std::min(layer_dependencies[i->second.end_layer_index], j->second.x);
    //             // last_layer_before_loading[j.second.x]
    //         }
    //     }
    // }
    for (auto i = memory_schedule.begin(); i != memory_schedule.end(); i++)
    {
        if (i->second.memory_type == 0)
        {
            last_layer_before_loading[i->second.start_layer_index] = std::max(last_layer_before_loading[i->second.start_layer_index], i->second.x - 1);
        }
    }

    for (int i = 0; i < get_layer_count(); i++)
    {
        if (last_layer_before_loading[i] < m_skip_layer_count) // < or <=?
            continue;                                          // doesn't matter
        layer_dependencies[last_layer_before_loading[i] - 1] = std::min(layer_dependencies[last_layer_before_loading[i] - 1], i);
    }

    // dependency sequence should be monotonic non-decreasing
    for (int i = get_layer_count() - 1; i > 0; i--)
    {
        layer_dependencies[i - 1] = std::min(layer_dependencies[i], layer_dependencies[i - 1]);
    }

    // dependency is invalid if one layer depends on next
    for (int i = 0; i < get_layer_count() - 1; i++)
    {
        if (layer_dependencies[i] == i + 1)
        {
            fprintf(stderr, "layer %d depends on next layer %d\n", i, i + 1);
            return -1;
        }
    }

    return 0;
}

int FlexnnSchedule::write_malloc_plan(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    fprintf(fp, "# weight_count blob_count intermediate_count (persistent_count)\n");
    fprintf(fp, "%d %d %d", m_weight_count, m_blob_count, m_intermediate_count);
    if (!m_persistent_offsets.empty())
    {
        fprintf(fp, " %d", (int)m_persistent_offsets.size());
    }
    fprintf(fp, "\n");
    fprintf(fp, "# weight_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[0].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[0][i]);
    }
    fprintf(fp, "# blob_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[1].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[1][i]);
    }
    fprintf(fp, "# intermediate_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[2].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[2][i]);
    }
    fprintf(fp, "# persistent_offsets\n");
    for (int i = 0; i < (int)m_persistent_offsets.size(); i++)
    {
        fprintf(fp, "%d\n", m_persistent_offsets[i]);
    }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::write_layer_dependencies(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    for (int i = 0; i < (int)m_layer_dependencies.size(); i++)
    {
        fprintf(fp, "%d\n", m_layer_dependencies[i]);
    }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::write_memory_layout(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    for (auto schedule : m_memory_schedule)
    {
        auto profile = schedule.second;
        // fprintf(fp, "%d,%d,%d,%d,%d,%d,%d\n", schedule.second.start_layer_index, schedule.second.end_layer_index, schedule.second.size, schedule.second.memory_type, schedule.second.malloc_count, schedule.second.x, schedule.second.y);
        fprintf(fp, "%d,%d,%d,%d,%d,%d\n", profile.x, profile.end_layer_index, profile.y, profile.size, profile.start_layer_index, profile.memory_type);
    }

    // // write persistent offsets, use the same format
    // for (int i = 0; i < (int)m_persistent_offsets.size(); i++)
    // {
    //     fprintf(fp, "%d\n", m_persistent_offsets[i]);
    // }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::generate_write_schedule(const char* malloc_plan_path, const char* layer_dependency_path, const char* memory_layout_path)
{
    if (!generate_malloc_plan(m_memory_schedule, m_malloc_plan))
        resolve_layer_dependencies(m_memory_schedule, m_layer_dependencies);

    if (!write_malloc_plan(malloc_plan_path))
        write_layer_dependencies(layer_dependency_path);

    if (memory_layout_path)
        write_memory_layout(memory_layout_path);

    return 0;
}

double FlexnnSchedule::predict_latency(const std::vector<int>& layer_dependencies)
{
    std::vector<double> loading_begin(get_layer_count(), .0f);
    std::vector<double> loading_end(get_layer_count(), .0f);
    std::vector<double> computing_begin(get_layer_count(), .0f);
    std::vector<double> computing_end(get_layer_count(), .0f);

    // layer 0: skip (Input)

    double tl = .0f, tc = .0f;
    loading_begin[m_skip_layer_count] = tl;
    tl += m_time_profiles[m_skip_layer_count].loading_duration;
    loading_end[m_skip_layer_count] = tl;

    for (int i = m_skip_layer_count; i < get_layer_count(); i++)
    {
        tc = std::max(loading_end[i], tc);
        computing_begin[i] = tc;
        tc += m_time_profiles[i].computing_duration;
        computing_end[i] = tc;

        // new loading tasks
        int start_index = layer_dependencies[i - 1];
        int end_index = layer_dependencies[i];
        for (int j = start_index; j < end_index; j++)
        {
            loading_begin[j] = tl;
            tl += m_time_profiles[j].loading_duration;
            loading_end[j] = tl;
        }
    }

    return tc;
}

int FlexnnSchedule::schedule_naive(int memory_budget)
{
    auto memory_profiles(m_memory_profiles);
    std::map<int, MemoryProfile> memory_schedule; // b.first=x=time, b.second=y=memory, sorted by x, for same x sorted by type and count (index)

    // find min peak memory
    std::vector<int> layer_memory(get_layer_count(), 0);
    std::vector<int> layer_weight_memory(get_layer_count(), 0);
    int total_weight_memory = 0;
    for (auto profile : memory_profiles)
    {
        for (int i = profile.second.start_layer_index; i <= profile.second.end_layer_index; i++)
        {
            layer_memory[i] += profile.second.size;
            if (profile.second.memory_type == 0)
            {
                layer_weight_memory[i] += profile.second.size;
                total_weight_memory += profile.second.size;
            }
        }
    }
    int peak_memory = 0, peak_index = -1;
    for (int i = 0; i < layer_memory.size(); i++)
    {
        if (layer_memory[i] >= peak_memory)
        {
            peak_memory = layer_memory[i];
            peak_index = i;
        }
    }

    // TODO: the ratio?
    int max_memory_margin = memory_budget - peak_memory;
    // dicide persistent weights within margin
    // TODO: how to select
    auto persistent_weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });
    // higher score means more likely to be selected as persistent.
    // near peak index, higher score.
    std::map<int, int> weight_scores;
    for (int i = 0; i < m_weight_count; i++)
    {
        auto profile = persistent_weight_it->second;
        int score = 0;
        if (profile.start_layer_index <= peak_index && profile.end_layer_index >= peak_index)
        {
            score = std::max(peak_index - profile.start_layer_index, profile.end_layer_index - peak_index);
        }
        else
        {
            score = std::min(std::abs(profile.start_layer_index - peak_index), std::abs(profile.end_layer_index - peak_index));
        }
        weight_scores.insert({persistent_weight_it->first, score});
        persistent_weight_it++;
    }
    // place higher score memory first, start from end of the buffe, don't exceed margin limit.
    // TODO: keep the order, important
    std::map<int, int> persistent_weights;
    // std::vector<int> persistent_offsets;
    int persistent_offset = alignOffsetSmall(memory_budget);
    int persistent_min_offset = alignOffsetBig(memory_budget - max_memory_margin);
    // greedy
    // TODO: optimize
    // only do this when it's IO bound and there are enough memory!
    if ((get_total_computing_duration() < 2 * get_total_loading_duration()) && (0.7 * (total_weight_memory - layer_weight_memory[peak_index]) < max_memory_margin))
    {
        for (auto it = weight_scores.begin(); it != weight_scores.end(); it++)
        {
            int size = memory_profiles[it->first].size;
            int next_offset = alignOffsetSmall(persistent_offset - size);
            if (next_offset < persistent_min_offset)
                continue;
            persistent_offset = next_offset;
            persistent_weights.insert({it->first, persistent_offset});

            // how to put them in all weights?
            // auto profile = memory_profiles[it->first];
            // profile.x = profile.start_layer_index;
            // profile.y = alignOffsetSmall(right - profile.size);
            // right = profile.y;

            // memory_schedule.insert({profile.memory_index(), profile});
            // memory_schedule.insert({it->first, memory_profiles[it->first]});

            // remove from memory_profiles
            // memory_profiles.erase(it->first);
        }
    }

    int dynamic_memory_budget = alignOffsetSmall(persistent_offset);

    // init xyplane
    init_xyplane(get_layer_count(), dynamic_memory_budget);

    // greedy schedule
    auto weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });
    auto blob_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 1;
    });
    auto intermediate_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 2;
    });

    int left = 0, right = dynamic_memory_budget;
    std::vector<int> lefts(get_layer_count(), 0), rights(get_layer_count(), dynamic_memory_budget);
    int layer_index = 0;
    // allocate blobs first, place blobs on two sides of the buffer
    for (int i = 0; i < m_blob_count; i++)
    {
        MemoryProfile profile = blob_it->second;

        // new layer
        if (profile.start_layer_index > layer_index)
        {
            int last_layer_index = layer_index;
            layer_index = profile.start_layer_index;
            // update left and right
            int next_left = 0, next_right = dynamic_memory_budget;
            for (auto schedule : memory_schedule)
            {
                // find not freed blobs
                if (schedule.second.end_layer_index >= layer_index)
                {
                    if (schedule.second.start_layer_index % 2 == 0)
                    {
                        next_left = std::max(next_left, schedule.second.y + schedule.second.size);
                    }
                    else
                    {
                        next_right = std::min(next_right, schedule.second.y);
                    }
                }
            }
            left = next_left;
            right = next_right;
        }

        // place blobs on two sides of the buffer
        if (layer_index % 2 == 0)
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetBig(left);
            left = profile.y + profile.size;
        }
        else
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetSmall(right - profile.size);
            right = profile.y;
        }

        memory_schedule.insert({profile.memory_index(), profile});
        // m_xy_plane->insert_xy(profile.x, profile.y, profile.size); // insert to xyplane
        m_xy_plane->insert_xrange_y(profile.start_layer_index, profile.end_layer_index, profile.y, profile.size); // insert to xyplane

        blob_it++;
    }

    // first allocate blobs, then weights and intermediates, try to preload weights.
    // place weights and intermediates in the middle of the buffer
    // first try: preload weights as much as possible, then schedule intermediates
    // fallback plan: don't preload at all, schedule weights and intermediates together

    // int offset = 0;
    // note that some weights are already scheduled as persistent weights and erased from memory_profiles
    int weight_count = 0;
    // int weight_count = persistent_weights.size();
    int intermediate_count = 0;
    int loading_x = 0;
    bool is_success = true;
    int max_preload_count = 50; // preload at most 10 layers
    for (int i = 0; i < get_layer_count(); i++)
    {
        // fprintf(stderr, "schedule layer %d.\n", i);
        is_success = true;
        m_xy_plane->backup();
        int weight_count_backup = weight_count;
        int intermediate_count_backup = intermediate_count;
        auto weight_it_backup = weight_it;
        auto intermediate_it_backup = intermediate_it;
        // auto l_xy_plane = m_xy_plane;
        // try preload
        while (weight_count < m_weight_count)
        {
            auto profile = weight_it->second;
            if (profile.start_layer_index > i)
                break;
            if (profile.memory_type != 0)
                break;

            if (persistent_weights.find(weight_it->first) != persistent_weights.end())
            {
                // already scheduled as persistent weights
                // dicide xy and add to memory schedule now
                profile.x = loading_x;
                profile.y = persistent_weights[weight_it->first];
                memory_schedule.insert({profile.memory_index(), profile});

                weight_count++;
                weight_it++;
                continue;
            }
            loading_x = std::max(loading_x, profile.start_layer_index - max_preload_count);
            auto ret = m_xy_plane->insert_xrange(loading_x, profile.end_layer_index, profile.size);
            if (ret.first < 0 || ret.second < 0)
            {
                fprintf(stderr, "preload schedule weight size %d failed, ret={%d, %d}.\n", profile.size, ret.first, ret.second);
                is_success = false;
                break;
            }
            // if (ret.first < profile.start_layer_index)
            // {
            //     fprintf(stderr, "preload %d at %d.\n", profile.start_layer_index, ret.first);
            // }
            profile.x = ret.first;
            profile.y = ret.second; // aligned
            memory_schedule.insert({profile.memory_index(), profile});

            loading_x = profile.x; // next loading starts not before x
            weight_count++;
            weight_it++;
        }

        if (!is_success)
        {
            m_xy_plane->save_payouts("xyplane.payout", i + 1);
            m_xy_plane->save_budgets("xyplane.budget", i + 1);
            break;
        }

        // schedule intermediates
        while (intermediate_count < m_intermediate_count)
        {
            auto profile = intermediate_it->second;
            if (profile.start_layer_index > i)
                break;
            if (profile.memory_type != 2)
                break;

            auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
            if (ret.first < 0 || ret.second < 0)
            {
                fprintf(stderr, "preload schedule intermediate size %d failed, ret={%d, %d}.\n", profile.size, ret.first, ret.second);
                is_success = false;
                break;
            }
            profile.x = ret.first;
            profile.y = ret.second; // aligned
            memory_schedule.insert({profile.memory_index(), profile});

            intermediate_count++;
            intermediate_it++;
        }

        if (!is_success)
        {
            // re-schedule this layer
            fprintf(stderr, "re-schedule layer %d.\n", i);
            m_xy_plane->restore();
            is_success = true;
            weight_count = weight_count_backup;
            intermediate_count = intermediate_count_backup;
            weight_it = weight_it_backup;
            intermediate_it = intermediate_it_backup;

            while (weight_count < m_weight_count)
            {
                auto profile = weight_it->second;
                if (profile.start_layer_index > i)
                    break;
                if (profile.memory_type != 0)
                    break;

                if (persistent_weights.find(weight_it->first) != persistent_weights.end())
                {
                    // already scheduled as persistent weights
                    // dicide xy and add to memory schedule now
                    profile.x = profile.start_layer_index;
                    profile.y = persistent_weights[weight_it->first];
                    memory_schedule.insert({profile.memory_index(), profile});

                    weight_count++;
                    weight_it++;
                    continue;
                }
                auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
                if (ret.first < 0 || ret.second < 0)
                {
                    fprintf(stderr, "re-schedule weight size %d failed, ret={%d, %d}.\n", profile.size, ret.first, ret.second);
                    is_success = false;
                    break;
                }
                // if (ret.first < profile.start_layer_index)
                // {
                //     fprintf(stderr, "preload %d at %d.\n", profile.start_layer_index, ret.first);
                // }
                profile.x = ret.first;
                profile.y = ret.second; // aligned
                memory_schedule.insert({profile.memory_index(), profile});

                loading_x = profile.x; // next loading starts not before x
                weight_count++;
                weight_it++;
            }

            if (!is_success)
            {
                m_xy_plane->save_payouts("xyplane.payout", i + 1);
                m_xy_plane->save_budgets("xyplane.budget", i + 1);
                break;
            }

            // schedule intermediates
            while (intermediate_count < m_intermediate_count)
            {
                auto profile = intermediate_it->second;
                if (profile.start_layer_index > i)
                    break;
                if (profile.memory_type != 2)
                    break;

                auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
                if (ret.first < 0 || ret.second < 0)
                {
                    fprintf(stderr, "re-schedule intermediate size %d failed, ret={%d, %d}.\n", profile.size, ret.first, ret.second);
                    is_success = false;
                    break;
                }
                profile.x = ret.first;
                profile.y = ret.second; // aligned
                memory_schedule.insert({profile.memory_index(), profile});

                intermediate_count++;
                intermediate_it++;
            }

            if (!is_success)
            {
                m_xy_plane->save_payouts("xyplane.payout", i + 1);
                m_xy_plane->save_budgets("xyplane.budget", i + 1);
                break;
            }
        }
    }

    m_xy_plane->save_payouts("xyplane.payout", get_layer_count());
    m_xy_plane->save_budgets("xyplane.budget", get_layer_count());

    if (!is_success)
    {
        fprintf(stderr, "schedule failed.\n");
        return -1;
    }

    // TODO: fallback plan

    // copy the schedule
    m_memory_schedule = memory_schedule;
    // offsets are persistent weights' offsets (values)
    m_persistent_offsets.clear();
    for (auto weight : persistent_weights)
    {
        m_persistent_offsets.push_back(weight.second);
    }

    return 0;
}

#endif // FLEXNN_SCHEDULE_H