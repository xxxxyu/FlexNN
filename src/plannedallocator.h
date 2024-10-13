#ifndef PLANNED_ALLOCATOR_H
#define PLANNED_ALLOCATOR_H

#include "allocator.h"

namespace flexnn {
class PlannedAllocator;
class PlannedAllocatorInterfacePrivate;
class NCNN_EXPORT PlannedAllocatorInterface : public ncnn::Allocator
{
public:
    PlannedAllocatorInterface();
    ~PlannedAllocatorInterface();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

    // memory_type = 0 for weight, 1 for blob, 2 for intermediate, -1 for don't change
    void set_attributes(int memory_type = -1);

    void set_allocator(PlannedAllocator* planned_allocator); // set the allocator that the interface belongs to

    PlannedAllocator* get_allocator() const;

    virtual int get_type() const;

private:
    PlannedAllocatorInterface(const PlannedAllocatorInterface&);
    PlannedAllocatorInterface& operator=(const PlannedAllocatorInterface&);

private:
    PlannedAllocatorInterfacePrivate* const d;
};

class PlannedAllocatorPrivate;
class PlannedAllocator
{
public:
    PlannedAllocator();
    ~PlannedAllocator();

    int init_buffer(size_t size);

    // memory_type = 0 for weight, 1 for blob, 2 for intermediate, -1 for don't change
    void* fastMalloc(int memory_type, size_t size);
    void fastFree(int memory_type, void* ptr);

    bool is_persistent(void* ptr) const;

    void add(PlannedAllocatorInterface* interface);

    void remove(PlannedAllocatorInterface* interface);

    void clear();

    void release_buffer();

    void load_malloc_plan(const char* path);

    void set_malloc_plan(const std::vector<std::vector<int> >& malloc_offsets, const std::vector<int>& persistent_offsets);

    // 0 for loading persistent weights, 1 for loading non-persistent weights. will affect the return value of is_persistent()
    void set_load_mode(int mode);

private:
    PlannedAllocator(const PlannedAllocator&);
    PlannedAllocator& operator=(const PlannedAllocator&);

private:
    PlannedAllocatorPrivate* const d;
};
} // namespace flexnn

#endif // PLANNED_ALLOCATOR_H