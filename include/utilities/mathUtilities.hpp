#ifndef _MATH_UTILITIES_HPP_

#define _MATH_UTILITIES_HPP_

#include <cmath>

namespace utilities
{
    template <typename T>
    bool compare(const T &a, const T &b, const T &tolerance)
    {
        return std::abs(a - b) < tolerance;
    }

    template <typename T>
    bool compare(const T &a, const T &b)
    {
        return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
    }

    template <typename T>
    int sign(const T &a)
    {
        if (compare(a, T(0)))
            return 0;
        else if (a > T(0))
            return 1;
        else
            return -1;
    }

}   // namespace utilities

#endif   // _MATH_UTILITIES_HPP_