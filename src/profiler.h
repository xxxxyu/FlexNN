#ifndef PROFILER_H
#define PROFILER_H

#include "allocator.h"
#include "flexnn_utils.h"
#include <map>

namespace flexnn {
//                      (Unified Records)
//                          Profiler
//      |----------------------|----------------------|
// Interface: weights   Interface: blobs    Interface: Intermediates
//      |                      |                      |
// fastMalloc()           fastMalloc()           fastMalloc()

// A profiler can have several interfaces that behave like different allocators, to be compatiable with NCNN design.
// The profiler collects the infomation from interfaces and summarize a profiling report.

class MemoryProfilerEvent
{
public:
    MemoryProfilerEvent()
        : layer_index(0), memory_type(0), event_type(0), ptr(0), size(0)
    {
        time = get_current_time();
    }

    MemoryProfilerEvent(int _layer_index, int _memory_type, int _event_type, void* _ptr, size_t _size)
        : layer_index(_layer_index), memory_type(_memory_type), event_type(_event_type), ptr(_ptr), size(_size)
    {
        time = get_current_time();
    }

public:
    int layer_index; // layer index
    int memory_type; // 0 for weight, 1 for blob, 2 for intermediate
    int event_type;  // 1 for malloc, 0 for free
    void* ptr;       // pointer to the memory
    size_t size;     // size of the memory, 0 for free
    double time;     // time of the event, automatically recorded in constructor
};

class MemoryProfilerInterfacePrivate;
class MemoryProfiler;
class NCNN_EXPORT MemoryProfilerInterface : public ncnn::Allocator
{
public:
    MemoryProfilerInterface();
    ~MemoryProfilerInterface();

    virtual void* fastMalloc(size_t size); // ask profiler for the memory
    virtual void fastFree(void* ptr);      // tell profiler to free the memory

    // memory_type = 0 for weight, 1 for blob, 2 for intermediate, -1 for don't change
    void set_attributes(int layer_index, int memory_type = -1);

    void set_profiler(MemoryProfiler* memory_profiler); // set the profiler that the interface belongs to

    virtual int get_type() const;
    // void clear(); // release all allocated memory

private:
    MemoryProfilerInterface(const MemoryProfilerInterface&);
    MemoryProfilerInterface& operator=(const MemoryProfilerInterface&);

private:
    MemoryProfilerInterfacePrivate* const d;
};

class MemoryProfilerPrivate;
class NCNN_EXPORT MemoryProfiler
{
public:
    MemoryProfiler();
    ~MemoryProfiler();

    void* fastMalloc(int layer_index, int memory_type, size_t size);
    void fastFree(int layer_index, int memory_type, void* ptr);

    // Add an interface to the profiler.
    // The profiler will not take the ownership of the interface.
    // The interface must be valid during the whole profiling process.
    void add(MemoryProfilerInterface* interface);

    // Remove an interface from the profiler.
    // The profiler will not release the interface.
    void remove(MemoryProfilerInterface* interface);

    // Clear all interfaces.
    // The profiler will not release the interfaces.
    void clear();

    // Print the profiling report.
    void print();

    // Save the profiling report to a csv file.
    void save(const char* csv_file);

    void save(std::vector<MemoryProfilerEvent>& events);

private:
    MemoryProfiler(const MemoryProfiler&);
    MemoryProfiler& operator=(const MemoryProfiler&);

private:
    MemoryProfilerPrivate* const d;
};

class LayerTimeProfile
{
public:
    LayerTimeProfile()
        : layer_index(0), loading_begin(0), loading_end(0), loading_duration(0), computing_begin(0), computing_end(0), computing_duration(0) {};

    LayerTimeProfile(int _layer_index, double _loading_begin, double _loading_end, double _computing_begin, double _computing_end)
        : layer_index(_layer_index), loading_begin(_loading_begin), loading_end(_loading_end), loading_duration(_loading_end - _loading_begin), computing_begin(_computing_begin), computing_end(_computing_end), computing_duration(_computing_end - _computing_begin) {};

public:
    int layer_index;
    double loading_begin;
    double loading_end;
    double loading_duration;
    double computing_begin;
    double computing_end;
    double computing_duration;
};

class TimeProfiler
{
public:
    virtual void insert(const LayerTimeProfile& profile) = 0;

    virtual void layer_loading_begin(int layer_index) = 0;
    virtual void layer_loading_end(int layer_index) = 0;
    virtual void layer_computing_begin(int layer_index) = 0;
    virtual void layer_computing_end(int layer_index) = 0;

    virtual void clear() = 0;
};

class UnlockedTimeProfilerPrivate;
class UnlockedTimeProfiler : public TimeProfiler
{
public:
    UnlockedTimeProfiler();
    ~UnlockedTimeProfiler();

    void insert(const LayerTimeProfile& profile);

    void layer_loading_begin(int layer_index);
    void layer_loading_end(int layer_index);
    void layer_computing_begin(int layer_index);
    void layer_computing_end(int layer_index);

    void clear();

    void print();

    void save(const char* csv_file);

    void save(std::vector<LayerTimeProfile>& profiles);

private:
    UnlockedTimeProfiler(const UnlockedTimeProfiler&);
    UnlockedTimeProfiler& operator=(const UnlockedTimeProfiler&);

private:
    UnlockedTimeProfilerPrivate* const d;
};

// locked time profiler for parallel run
class LockedTimeProfilerPrivate;
class LockedTimeProfiler : public TimeProfiler
{
public:
    LockedTimeProfiler();
    ~LockedTimeProfiler();

    void insert(const LayerTimeProfile& profile);

    void layer_loading_begin(int layer_index);
    void layer_loading_end(int layer_index);
    void layer_computing_begin(int layer_index);
    void layer_computing_end(int layer_index);

    void clear();

    void print();

    void save(const char* csv_file);

private:
    LockedTimeProfiler(const LockedTimeProfiler&);
    LockedTimeProfiler& operator=(const LockedTimeProfiler&);

private:
    LockedTimeProfilerPrivate* const d;
};

} // namespace flexnn

#endif // PROFILER_H