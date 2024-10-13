#include "plannedallocator.h"
#include <set>

namespace flexnn {

class PlannedAllocatorInterfacePrivate
{
public:
    int layer_index; // layer index
    int memory_type; // 0 for weight, 1 for blob, 2 for intermediate
    int count;       // count this interface's allocations in an inference session
public:
    PlannedAllocator* allocator;
};

PlannedAllocatorInterface::PlannedAllocatorInterface()
    : d(new PlannedAllocatorInterfacePrivate)
{
    d->layer_index = 0;
    d->memory_type = 0;
    d->count = 0;
}

PlannedAllocatorInterface::~PlannedAllocatorInterface()
{
}

PlannedAllocatorInterface::PlannedAllocatorInterface(const PlannedAllocatorInterface&)
    : d(0)
{
}

PlannedAllocatorInterface& PlannedAllocatorInterface::operator=(const PlannedAllocatorInterface&)
{
    return *this;
}

void* PlannedAllocatorInterface::fastMalloc(size_t size)
{
    return d->allocator->fastMalloc(d->memory_type, size);
}

void PlannedAllocatorInterface::fastFree(void* ptr)
{
    d->allocator->fastFree(d->memory_type, ptr);
}

void PlannedAllocatorInterface::set_attributes(int memory_type)
{
    if (memory_type != -1)
    {
        d->memory_type = memory_type;
    }
}

void PlannedAllocatorInterface::set_allocator(PlannedAllocator* allocator)
{
    d->allocator = allocator;
}

PlannedAllocator* PlannedAllocatorInterface::get_allocator() const
{
    return d->allocator;
}

int PlannedAllocatorInterface::get_type() const
{
    return 4;
}

class PlannedAllocatorPrivate
{
public:
    void* buffer;                                       // unified buffer for all allocations
    size_t buffer_size;                                 // total buffer size
    std::vector<PlannedAllocatorInterface*> interfaces; // vector of interfaces
    std::vector<std::vector<void*> > allocations;       // allocations[memory_type][count]
    std::vector<int> counters;                          // counters[memory_type]
    std::set<void*> persistent_weights;                 // persistent weights
    int load_mode;
    ncnn::Mutex lock;
};

PlannedAllocator::PlannedAllocator()
    : d(new PlannedAllocatorPrivate)
{
    d->buffer = 0;
    d->allocations.resize(3);
    d->counters.resize(3, 0);
    d->persistent_weights.clear();
    d->load_mode = -1;
}

PlannedAllocator::~PlannedAllocator()
{
    if (d->buffer)
    {
        ncnn::fastFree(d->buffer);
    }
    delete d;
}

int PlannedAllocator::init_buffer(size_t size)
{
    fprintf(stderr, "PlannedAllocator::init_buffer %ld\n", size);

    if (d->buffer)
    {
        ncnn::fastFree(d->buffer);
    }

    d->buffer = ncnn::fastMalloc(size);
    if (!d->buffer)
    {
        NCNN_LOGE("PlannedAllocator::PlannedAllocator() failed to allocate %ld bytes", size);
        return -1;
    }
    d->buffer_size = size;
    return 0;
}

PlannedAllocator::PlannedAllocator(const PlannedAllocator&)
    : d(0)
{
}

PlannedAllocator& PlannedAllocator::operator=(const PlannedAllocator&)
{
    return *this;
}

void* PlannedAllocator::fastMalloc(int memory_type, size_t size)
{
    d->lock.lock();
    if (d->counters[memory_type] >= (int)d->allocations[memory_type].size())
    {
        NCNN_LOGE("PlannedAllocator::fastMalloc() failed to allocate %d bytes from memory type %d", (int)size, memory_type);
        return 0;
    }
    void* ptr = d->allocations[memory_type][d->counters[memory_type]];
    d->counters[memory_type]++;
    d->lock.unlock();
    return ptr;
}

void PlannedAllocator::fastFree(int /*memory_type*/, void* /*ptr*/)
{
    // do nothing
}

void PlannedAllocator::add(PlannedAllocatorInterface* interface)
{
    interface->set_allocator(this);
    d->interfaces.push_back(interface);
}

void PlannedAllocator::remove(PlannedAllocatorInterface* interface)
{
    std::vector<PlannedAllocatorInterface*>::iterator it = std::find(d->interfaces.begin(), d->interfaces.end(), interface);
    if (it != d->interfaces.end())
    {
        d->interfaces.erase(it);
    }
}

void PlannedAllocator::clear()
{
    for (int i = 0; i < 3; i++)
    {
        d->counters[i] = 0;
    }
}

void PlannedAllocator::load_malloc_plan(const char* path)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        NCNN_LOGE("PlannedAllocator::load_malloc_plan() failed to open %s", path);
        return;
    }

    char line[256];
    int weight_count = 0;
    int blob_count = 0;
    int intermediate_count = 0;
    int persistent_count = 0;
    int line_count = 0;
    while (fgets(line, 256, fp))
    {
        if (line[0] == '#')
        {
            continue;
        }
        if (line_count == 0)
        {
            int ret = sscanf(line, "%d %d %d %d", &weight_count, &blob_count, &intermediate_count, &persistent_count);
            line_count++;
            if (ret == 3)
            {
                persistent_count = 0;
            }
            if (ret < 3)
            {
                NCNN_LOGE("PlannedAllocator::load_malloc_plan() parse first line failed.");
                return;
            }
            continue;
        }

        int offset = 0;
        sscanf(line, "%d", &offset);

        if (offset < 0)
        {
            NCNN_LOGE("PlannedAllocator::load_malloc_plan() invalid offset %d", offset);
            continue;
        }

        if (line_count <= weight_count)
        {
            d->allocations[0].push_back((char*)d->buffer + offset);
        }
        else if (line_count <= weight_count + blob_count)
        {
            d->allocations[1].push_back((char*)d->buffer + offset);
        }
        else if (line_count <= weight_count + blob_count + intermediate_count)
        {
            d->allocations[2].push_back((char*)d->buffer + offset);
        }
        else if (line_count <= weight_count + blob_count + intermediate_count + persistent_count)
        {
            d->persistent_weights.insert((char*)d->buffer + offset);
        }
        else
        {
            NCNN_LOGE("PlannedAllocator::load_malloc_plan() invalid line count %d\n", line_count);
            continue;
        }

        line_count++;
    }

    // report count read
    NCNN_LOGE("PlannedAllocator::load_malloc_plan() weight_count %d blob_count %d intermediate_count %d persistent_count %d", weight_count, blob_count, intermediate_count, persistent_count);

    return;
}

void PlannedAllocator::set_malloc_plan(const std::vector<std::vector<int> >& malloc_offsets, const std::vector<int>& persistent_offsets)
{
    // fprintf(stderr, "PlannedAllocator::set_malloc_plan\n");
    // fprintf(stderr, "malloc_offsets.size() = %d\n", (int)malloc_offsets.size());
    d->lock.lock();
    // copy malloc plan and persistent weights, ptr = buffer + offset
    for (int i = 0; i < 3; i++)
    {
        // fprintf(stderr, "malloc_offsets[%d].size() = %d\n", i, (int)malloc_offsets[i].size());
        d->allocations[i].clear();
        for (size_t j = 0; j < malloc_offsets[i].size(); j++)
        {
            d->allocations[i].push_back((char*)d->buffer + malloc_offsets[i][j]);
        }
    }
    fprintf(stderr, "read malloc plan, weight %d, blob %d, intermediate %d\n", d->allocations[0].size(), d->allocations[1].size(), d->allocations[2].size());
    d->persistent_weights.clear();
    for (auto it = persistent_offsets.begin(); it != persistent_offsets.end(); it++)
    {
        d->persistent_weights.insert((char*)d->buffer + *it);
    }
    fprintf(stderr, "read persistent weights, %d\n", d->persistent_weights.size());
    d->lock.unlock();
}

bool PlannedAllocator::is_persistent(void* ptr) const
{
    bool ret = d->persistent_weights.find(ptr) != d->persistent_weights.end();
    if (d->load_mode == 0)
    {
        return !ret;
    }
    if (d->load_mode == 1)
    {
        return ret;
    }

    NCNN_LOGE("PlannedAllocator::is_persistent() invalid load_mode %d", d->load_mode);
    return false;
}

void PlannedAllocator::set_load_mode(int mode)
{
    d->load_mode = mode;
}

void PlannedAllocator::release_buffer()
{
    if (d->buffer)
    {
        ncnn::fastFree(d->buffer);
        d->buffer = 0;
    }
}

} // namespace flexnn
