#ifndef XY_PLANE_H
#define XY_PLANE_H

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

#include "stdio.h"

// NCNN
// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSizeBase(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

// return >= sz
static inline size_t alignSizeBig(size_t sz, int n)
{
    return alignSizeBase(sz, n);
}

// return <= sz
static inline size_t alignSizeSmall(size_t sz, int n)
{
    return alignSizeBase(sz - n + 1, n);
}

class XYPlane
{
public:
    XYPlane(int x, int y, int _align)
        : x0(x), y0(alignSizeSmall(y, _align)), align(_align)
    {
        total_budgets.resize(x0, std::list<std::pair<int, int> >(1, {0, y0})); // initialize budget 0 ~ y
        total_payouts.resize(x0, std::list<std::pair<int, int> >());
    }

    // find available budgets for given x, dy. return a list of pointers to the budgets, or empty list if failed.
    std::list<std::pair<int, int> > find_budgets(int x, int dy)
    {
        std::list<std::pair<int, int> > ret;
        dy = align_big(dy);

        auto& budget = total_budgets[x];

        // find budgets
        auto it = budget.begin();
        for (; it != budget.end(); ++it)
        {
            if (it->second >= dy)
            {
                ret.push_back(*it);
            }
        }

        return ret;
    }

    // find one available budget
    std::pair<int, int> find_budget(int x, int dy, int opt = FIND_FIRST)
    {
        dy = align_big(dy);
        return find_budget_yrange(x, 0, y0, dy, opt);
    }

    // find available budgets for given x, dy, containing (y, dy) such that y>=y1 and y+dy<=y2. return a list of pointers to the budgets, or empty list if failed.
    std::list<std::pair<int, int> > find_budgets_yrange(int x, int y1, int y2, int dy)
    {
        std::list<std::pair<int, int> > ret;
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        // range too small
        if (y2 - y1 < dy)
        {
            return ret;
        }

        auto& budget = total_budgets[x];

        // find budgets
        auto it = budget.begin();
        for (; it != budget.end(); ++it)
        {
            if (it->second >= dy && it->first + dy <= y2 && y1 + dy <= it->first + it->second)
            {
                ret.push_back(*it);
            }
        }

        return ret;
    }

    // find one budget
    std::pair<int, int> find_budget_yrange(int x, int y1, int y2, int dy, int opt = FIND_FIRST)
    {
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        // range too small
        if (y2 - y1 < dy)
        {
            return {-10, -10};
        }

        auto& budget = total_budgets[x];

        // find a budget
        if (opt == FIND_FIRST)
        {
            auto it = budget.begin();
            for (; it != budget.end(); ++it)
            {
                if (it->second >= dy && it->first + dy <= y2 && y1 + dy <= it->first + it->second)
                {
                    return *it;
                }
            }
        }
        else if (opt == FIND_SMALLEST_Y)
        {
            auto it = budget.begin();
            std::pair<int, int> ret = {-1, -1};
            for (; it != budget.end(); ++it)
            {
                if (it->second >= dy && it->first + dy <= y2 && y1 + dy <= it->first + it->second)
                {
                    if (ret.first > it->first)
                    {
                        ret = *it;
                    }
                }
            }
            return ret;
        }
        else if (opt == FIND_LARGEST_Y)
        {
            auto it = budget.begin();
            std::pair<int, int> ret = {-1, -1};
            for (; it != budget.end(); ++it)
            {
                if (it->second >= dy && it->first + dy <= y2 && y1 + dy <= it->first + it->second)
                {
                    if (ret.first < it->first)
                    {
                        ret = *it;
                    }
                }
            }
            return ret;
        }

        return {-1, -1};
    }

    // find available budgets for given x, dy, containing (y, dy) such that y>=y_min and y+dy<=y_max.
    // where y_min is the maximum y of the given budgets and y1, and y_max is the minimum y+dy of the given budgets and y2.
    // return a list of pointers to the budgets, or empty list if failed.
    std::list<std::pair<int, int> > find_budgets_yrange(const std::list<std::pair<int, int> >& prev_budgets, int x, int y1, int y2, int dy)
    {
        std::list<std::pair<int, int> > ret;
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        // range to small
        if (y2 - y1 < dy)
        {
            return ret;
        }

        // find y_min and y_max
        int y_min = y1;
        int y_max = y2;
        for (auto it = prev_budgets.begin(); it != prev_budgets.end(); ++it)
        {
            y_min = std::max(y_min, (*it).first);
            y_max = std::min(y_max, (*it).first + (*it).second);
        }

        // find a budget
        auto& budget = total_budgets[x];

        auto it = budget.begin();
        for (; it != budget.end(); ++it)
        {
            // too small
            if (it->second < dy)
            {
                continue;
            }
            // out of range: right
            if (it->first + dy > y_max)
            {
                continue;
            }
            // out of range: left
            if (it->first + it->second < y_min + dy)
            {
                continue;
            }
            // found
            ret.push_back(*it);
        }

        return ret;
    }

    // TODO: one budget. BUT not necessary.

    // recursively find budgets for given x1, x2, y1, y2, dy. add new found budgets to found_budgets.
    // only keep lists with maximum size in the list.
    // list is organized in reversed order. i.e. the first element is x2's budget.
    // return the maximum size of lists found. success if return x2-x1+1.
    int find_budgets_xrange_yrange(std::list<std::list<std::pair<int, int> > >& found_budgets, int x1, int x2, int y1, int y2, int dy)
    {
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        // terminate recursive calls
        if (x1 == x2)
        {
            for (auto budget : find_budgets_yrange(x1, y1, y2, dy))
            {
                found_budgets.push_back(std::list<std::pair<int, int> >(1, budget));
            }
            return 1;
        }

        // find budgets for x1 based on [x1+1, x2]'s found_budgets
        int ret = find_budgets_xrange_yrange(found_budgets, x1 + 1, x2, y1, y2, dy);

        // return if [x1+1, x2] already failed
        if (ret < x2 - x1)
        {
            return ret;
        }

        // go on if success
        std::list<std::list<std::pair<int, int> > > new_found_budgets;
        for (auto vec : found_budgets)
        {
            // find budgets for x1 based on [x1+1, x2]'s found_budgets
            auto budgets = find_budgets_yrange(vec, x1, y1, y2, dy);
            for (auto budget : budgets)
            {
                vec.push_back(budget);
                new_found_budgets.push_back(vec);
                vec.pop_back();
            }
        }
        // return if failed
        if (new_found_budgets.empty())
        {
            return ret;
        }
        // update found_budgets if success
        found_budgets = new_found_budgets;
        return ret + 1;
    }

    // find continuous available budgets for given [x1, x2], dy.
    // return a list of lists of pointers to the budgets with max possible length (might be shorter than x2-x1+1).
    // list is organized in reversed order. i.e. the first element is x2's budget.
    std::list<std::list<std::pair<int, int> > > find_budgets_xrange(int x1, int x2, int dy)
    {
        std::list<std::list<std::pair<int, int> > > ret;
        dy = align_big(dy);

        // no y limit
        find_budgets_xrange_yrange(ret, x1, x2, 0, y0, dy);

        return ret;
    }

    // insert (y, dy) to x, return y if success, -1 if failed
    int insert_xy(int x, int y, int dy)
    {
        if (!is_aligned(y))
        {
            return -2;
        }

        dy = align_big(dy);

        auto& budget = total_budgets[x];
        auto& payout = total_payouts[x];

        // find a budget
        auto it = budget.begin();
        for (; it != budget.end(); ++it)
        {
            if ((it->first <= y) && (it->first + it->second >= y + dy))
            {
                // insert to total_payouts
                payout.push_back({y, dy});
                // fprintf(stderr, "insert xy: %d %d %d\n", x, y, dy);

                // update budgets
                if (it->first < y)
                {
                    budget.push_back({it->first, y - it->first});
                }
                if (it->first + it->second > y + dy)
                {
                    budget.push_back({y + dy, it->first + it->second - y - dy});
                }
                budget.erase(it);

                // end
                return y;
            }
        }

        // failed
        return -5;
    }

    // insert dy to x, return y if success, -1 if failed
    int insert_x_yrange(int x, int y1, int y2, int dy)
    {
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        auto found_budgets = find_budgets_yrange(x, y1, y2, dy);

        // TODO: how to select a budget
        // now is a greedy algorithm: select the first one

        auto it = found_budgets.begin();
        for (; it != found_budgets.end(); ++it)
        {
            auto& budget = *it;
            int y = std::max(budget.first, y1);
            return insert_xy(x, y, dy);
        }

        // failed
        return -6;
    }

    // insert (y,dy) to [x1,x2], return y if success, -1 if failed
    int insert_xrange_y(int x1, int x2, int y, int dy)
    {
        if (!is_aligned(y))
        {
            return -10;
        }
        dy = align_big(dy);

        // fprintf(stderr, "insert x range: %d ~ %d.\n", x1, x2);
        for (int x = x1; x <= x2; ++x)
        {
            int ret = insert_xy(x, y, dy);
            if (ret != y)
            {
                // failed
                // cannot rollback, make sure you have found valid budgets!
                return ret;
            }
        }
        return y;
    }

    // insert (y,dy) to [(x1),x2] in [y1,y2], return (start_x,y) if success, -1 if failed.
    // insert as many x as possible.
    // TODO: how to handle found length <= x2-x1+1.
    std::pair<int, int> insert_xrange_yrange(int x1, int x2, int y1, int y2, int dy)
    {
        dy = align_big(dy);
        y1 = align_big(y1);
        y2 = align_small(y2);

        std::list<std::list<std::pair<int, int> > > found_budgets;
        find_budgets_xrange_yrange(found_budgets, x1, x2, y1, y2, dy);
        // not found
        if (found_budgets.empty())
        {
            return {-11, -11};
        }
        // found length <= x2-x1+1
        int found_length = found_budgets.front().size();
        int start_x = x2 - found_length + 1;
        // found, the insert
        // calculate the y, TODO: algorithm
        // now: select minimum y in the first list of found_budgets
        int y = y1;
        for (auto budgets : found_budgets)
        {
            for (auto budget : budgets)
            {
                if (budget.first > y)
                {
                    y = budget.first;
                }
            }
            break; // TODO: how to select a proper y?
        }
        return {start_x, insert_xrange_y(start_x, x2, y, dy)};
    }

    std::pair<int, int> insert_xrange(int x1, int x2, int dy)
    {
        dy = align_big(dy);
        // fprintf(stderr, "insert x range: %d %d %d.\n", x1, x2, dy);
        return insert_xrange_yrange(x1, x2, 0, y0, dy);
    }

public:
    std::vector<std::list<std::pair<int, int> > > total_budgets;  // total_budgets[x]={(y,dy), ...}.
    std::vector<std::list<std::pair<int, int> > > total_payouts;  // total_payouts[x]={(y,dy), ...}.
    std::vector<std::list<std::pair<int, int> > > backup_budgets; // backup_budgets[x]={(y,dy), ...}.
    std::vector<std::list<std::pair<int, int> > > backup_payouts; // backup_payouts[x]={(y,dy), ...}.
    int y0;
    int x0;
    int align; // all budgets and payouts are aligned to this number (both offset and size)

public:
    // utils
    bool is_aligned(size_t x) const
    {
        return (x % align) == 0;
    }

    size_t align_big(size_t x) const
    {
        return alignSizeBig(x, align);
    }

    size_t align_small(size_t x) const
    {
        return alignSizeSmall(x, align);
    }

    // save all payouts on the plane to a text file
    void save_payouts(const char* path, int end_x = 0)
    {
        FILE* fp = fopen(path, "w");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", path);
            return;
        }

        for (int x = 0; x < (end_x > 0 ? end_x : x0); ++x)
        {
            for (auto& payout : total_payouts[x])
            {
                fprintf(fp, "%d,%d,%d\n", x, payout.first, payout.second);
            }
        }

        fclose(fp);
    }

    // save all budgets on the plane to a text file
    void save_budgets(const char* path, int end_x = 0)
    {
        FILE* fp = fopen(path, "w");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", path);
            return;
        }

        for (int x = 0; x < (end_x > 0 ? end_x : x0); ++x)
        {
            for (auto& budget : total_budgets[x])
            {
                fprintf(fp, "%d,%d,%d\n", x, budget.first, budget.second);
            }
        }

        fclose(fp);
    }

    void backup()
    {
        backup_budgets = total_budgets;
        backup_payouts = total_payouts;
    }

    void restore()
    {
        total_budgets = backup_budgets;
        total_payouts = backup_payouts;
    }

public:
    // constants for parameters
    static const int FIND_FIRST = 0;
    static const int FIND_SMALLEST_Y = 100;
    static const int FIND_LARGEST_Y = 101;
};

#endif // XY_PLANE_H