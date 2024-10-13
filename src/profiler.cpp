
#include "profiler.h"
#include "platform.h"

namespace flexnn {
class MemoryProfilerInterfacePrivate
{
public:
    int layer_index; // layer index
    int memory_type; // 0 for weight, 1 for blob, 2 for intermediate
    int count;

public:
    MemoryProfiler* memory_profiler;
};

MemoryProfilerInterface::MemoryProfilerInterface()
    : d(new MemoryProfilerInterfacePrivate)
{
    d->layer_index = 0;
    d->memory_type = 0;
    d->count = 0;
}

MemoryProfilerInterface::~MemoryProfilerInterface()
{
}

MemoryProfilerInterface::MemoryProfilerInterface(const MemoryProfilerInterface&)
    : d(0)
{
}

MemoryProfilerInterface& MemoryProfilerInterface::operator=(const MemoryProfilerInterface&)
{
    return *this;
}

void* MemoryProfilerInterface::fastMalloc(size_t size)
{
    return d->memory_profiler->fastMalloc(d->layer_index, d->memory_type, size);
}

void MemoryProfilerInterface::fastFree(void* ptr)
{
    d->memory_profiler->fastFree(d->layer_index, d->memory_type, ptr);
}

int MemoryProfilerInterface::get_type() const
{
    return 3;
}

void MemoryProfilerInterface::set_attributes(int layer_index, int memory_type)
{
    d->layer_index = layer_index;
    if (memory_type != -1)
    {
        d->memory_type = memory_type;
    }
}

void MemoryProfilerInterface::set_profiler(MemoryProfiler* memory_profiler)
{
    d->memory_profiler = memory_profiler;
}

class MemoryProfilerPrivate
{
public:
    std::vector<MemoryProfilerInterface*> interfaces;

public:
    std::vector<MemoryProfilerEvent> events;
    ncnn::Mutex lock;
};

MemoryProfiler::MemoryProfiler()
    : d(new MemoryProfilerPrivate)
{
}

MemoryProfiler::~MemoryProfiler()
{
    clear();

    delete d;
}

MemoryProfiler::MemoryProfiler(const MemoryProfiler&)
    : d(0)
{
}

MemoryProfiler& MemoryProfiler::operator=(const MemoryProfiler&)
{
    return *this;
}

void* MemoryProfiler::fastMalloc(int layer_index, int memory_type, size_t size)
{
    d->lock.lock();

    void* ptr = ncnn::fastMalloc(size); // actual size = size + NCNN_MALLOC_OVERREAD

    d->events.push_back(MemoryProfilerEvent(layer_index, memory_type, 1, ptr, size + NCNN_MALLOC_OVERREAD));

    // NCNN_LOGE("MemoryProfiler fastMalloc %p %zu", ptr, size);
    d->lock.unlock();

    return ptr;
}

void MemoryProfiler::fastFree(int layer_index, int memory_type, void* ptr)
{
    d->lock.lock();

    d->events.push_back(MemoryProfilerEvent(layer_index, memory_type, 0, ptr, 0));

    // NCNN_LOGE("MemoryProfiler fastFree %p", ptr);

    ncnn::fastFree(ptr);

    d->lock.unlock();

    return;
}

void MemoryProfiler::add(MemoryProfilerInterface* memory_profiler_interface)
{
    memory_profiler_interface->set_profiler(this);

    d->interfaces.push_back(memory_profiler_interface);
}

void MemoryProfiler::remove(MemoryProfilerInterface* memory_profiler_interface)
{
    std::vector<MemoryProfilerInterface*>::iterator it = std::find(d->interfaces.begin(), d->interfaces.end(), memory_profiler_interface);
    if (it != d->interfaces.end())
        d->interfaces.erase(it);
}

void MemoryProfiler::clear()
{
    d->lock.lock();
    d->events.clear();
    d->interfaces.clear();
    d->lock.unlock();
}

void MemoryProfiler::print()
{
    // TODO
    return;
}

void MemoryProfiler::save(const char* csv_file)
{
    FILE* fp = fopen(csv_file, "w");
    if (!fp)
    {
        NCNN_LOGE("MemoryProfiler save %s failed", csv_file);
        return;
    }

    fprintf(fp, "layer_index,memory_type,event_type,ptr,size,time\n");

    for (size_t i = 0; i < d->events.size(); i++)
    {
        const MemoryProfilerEvent& event = d->events[i];

        fprintf(fp, "%d,%d,%d,%p,%zu,%f\n", event.layer_index, event.memory_type, event.event_type, event.ptr, event.size, event.time);
    }

    fclose(fp);
    fprintf(stderr, "Saving profiling results success.\n");
}

void MemoryProfiler::save(std::vector<MemoryProfilerEvent>& events)
{
    events = d->events;
}

class UnlockedTimeProfilerPrivate
{
public:
    std::map<int, LayerTimeProfile> profiles;
};

UnlockedTimeProfiler::UnlockedTimeProfiler()
    : d(new UnlockedTimeProfilerPrivate)
{
}

UnlockedTimeProfiler::~UnlockedTimeProfiler()
{
    clear();

    delete d;
}

UnlockedTimeProfiler::UnlockedTimeProfiler(const UnlockedTimeProfiler&)
    : d(0)
{
}

UnlockedTimeProfiler& UnlockedTimeProfiler::operator=(const UnlockedTimeProfiler&)
{
    return *this;
}

void UnlockedTimeProfiler::insert(const LayerTimeProfile& profile)
{
    d->profiles.insert(std::pair<int, LayerTimeProfile>(profile.layer_index, profile));
}

void UnlockedTimeProfiler::layer_loading_begin(int layer_index)
{
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].loading_begin = get_current_time();
}

void UnlockedTimeProfiler::layer_loading_end(int layer_index)
{
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].loading_end = get_current_time();
    if (d->profiles[layer_index].loading_begin > 0)
    {
        d->profiles[layer_index].loading_duration = d->profiles[layer_index].loading_end - d->profiles[layer_index].loading_begin;
    }
}

void UnlockedTimeProfiler::layer_computing_begin(int layer_index)
{
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].computing_begin = get_current_time();
}

void UnlockedTimeProfiler::layer_computing_end(int layer_index)
{
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].computing_end = get_current_time();
    if (d->profiles[layer_index].computing_begin > 0)
    {
        d->profiles[layer_index].computing_duration = d->profiles[layer_index].computing_end - d->profiles[layer_index].computing_begin;
    }
}

void UnlockedTimeProfiler::clear()
{
    d->profiles.clear();
}

void UnlockedTimeProfiler::print()
{
    // TODO
    return;
}

void UnlockedTimeProfiler::save(const char* csv_file)
{
    FILE* fp = fopen(csv_file, "w");
    if (!fp)
    {
        NCNN_LOGE("UnlockedTimeProfiler save %s failed", csv_file);
        return;
    }

    fprintf(fp, "layer_index,loading_begin,loading_end,loading_duration,computing_begin,computing_end,computing_duration\n");

    for (std::map<int, LayerTimeProfile>::iterator it = d->profiles.begin(); it != d->profiles.end(); it++)
    {
        const LayerTimeProfile& profile = it->second;

        fprintf(fp, "%d,%f,%f,%f,%f,%f,%f\n", profile.layer_index, profile.loading_begin, profile.loading_end, profile.loading_duration, profile.computing_begin, profile.computing_end, profile.computing_duration);
    }

    fclose(fp);
    fprintf(stderr, "Saving profiling results success.\n");
}

void UnlockedTimeProfiler::save(std::vector<LayerTimeProfile>& profiles)
{
    profiles.clear();
    for (std::map<int, LayerTimeProfile>::iterator it = d->profiles.begin(); it != d->profiles.end(); it++)
    {
        profiles.push_back(it->second);
    }
}

class LockedTimeProfilerPrivate
{
public:
    std::map<int, LayerTimeProfile> profiles;
    ncnn::Mutex lock;
};

LockedTimeProfiler::LockedTimeProfiler()
    : d(new LockedTimeProfilerPrivate)
{
}

LockedTimeProfiler::~LockedTimeProfiler()
{
    clear();

    delete d;
}

LockedTimeProfiler::LockedTimeProfiler(const LockedTimeProfiler&)
    : d(0)
{
}

LockedTimeProfiler& LockedTimeProfiler::operator=(const LockedTimeProfiler&)
{
    return *this;
}

void LockedTimeProfiler::insert(const LayerTimeProfile& profile)
{
    d->lock.lock();
    d->profiles.insert(std::pair<int, LayerTimeProfile>(profile.layer_index, profile));
    d->lock.unlock();
}

void LockedTimeProfiler::layer_loading_begin(int layer_index)
{
    d->lock.lock();
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].loading_begin = get_current_time();
    d->lock.unlock();
}

void LockedTimeProfiler::layer_loading_end(int layer_index)
{
    d->lock.lock();
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].loading_end = get_current_time();
    if (d->profiles[layer_index].loading_begin > 0)
    {
        d->profiles[layer_index].loading_duration = d->profiles[layer_index].loading_end - d->profiles[layer_index].loading_begin;
    }
    d->lock.unlock();
}

void LockedTimeProfiler::layer_computing_begin(int layer_index)
{
    d->lock.lock();
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].computing_begin = get_current_time();
    d->lock.unlock();
}

void LockedTimeProfiler::layer_computing_end(int layer_index)
{
    d->lock.lock();
    d->profiles[layer_index].layer_index = layer_index;
    d->profiles[layer_index].computing_end = get_current_time();
    if (d->profiles[layer_index].computing_begin > 0)
    {
        d->profiles[layer_index].computing_duration = d->profiles[layer_index].computing_end - d->profiles[layer_index].computing_begin;
    }
    d->lock.unlock();
}

void LockedTimeProfiler::clear()
{
    d->lock.lock();
    d->profiles.clear();
    d->lock.unlock();
}

void LockedTimeProfiler::print()
{
    // TODO
    return;
}

void LockedTimeProfiler::save(const char* csv_file)
{
    FILE* fp = fopen(csv_file, "w");
    if (!fp)
    {
        NCNN_LOGE("LockedTimeProfiler save %s failed", csv_file);
        return;
    }

    fprintf(fp, "layer_index,loading_begin,loading_end,loading_duration,computing_begin,computing_end,computing_duration\n");

    d->lock.lock();
    for (std::map<int, LayerTimeProfile>::iterator it = d->profiles.begin(); it != d->profiles.end(); it++)
    {
        const LayerTimeProfile& profile = it->second;

        fprintf(fp, "%d,%f,%f,%f,%f,%f,%f\n", profile.layer_index, profile.loading_begin, profile.loading_end, profile.loading_duration, profile.computing_begin, profile.computing_end, profile.computing_duration);
    }
    d->lock.unlock();

    fclose(fp);
    fprintf(stderr, "Saving profiling results success.\n");
}

// --
} // namespace flexnn