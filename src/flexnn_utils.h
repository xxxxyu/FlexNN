#ifndef FLEXNN_UTILS_H
#define FLEXNN_UTILS_H

#include "stdio.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

namespace flexnn {

template<typename T>
void print_vector(const std::vector<T>& vec)
{
    std::cout << "[ ";
    for (const auto& elem : vec)
    {
        std::cout << elem << ' ';
    }
    std::cout << "]\n";
}

static double get_current_time()
{
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
}

} // namespace flexnn

#endif // FLEXNN_UTILS_H
