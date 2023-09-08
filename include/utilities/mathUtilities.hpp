#ifndef _MATH_UTILITIES_HPP_

#define _MATH_UTILITIES_HPP_

#include "vector3d.hpp"

#include <cmath>     // for fabs
#include <cstdlib>   // for abs
#include <limits>    // for numeric_limits

namespace utilities
{
    /**
     * @brief compares two numbers with a tolerance
     *
     * @tparam T
     * @param a
     * @param b
     * @param tolerance
     * @return true
     * @return false
     */
    template <typename T>
    bool compare(const T &a, const T &b, const T &tolerance)
    {
        return std::abs(a - b) < tolerance;
    }

    bool compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b, const double &tolerance);

    /**
     * @brief compares two numbers via machine precision
     *
     * @tparam T
     * @param a
     * @param b
     * @return true
     * @return false
     */
    template <typename T>
    bool compare(const T &a, const T &b)
    {
        return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
    }

    bool compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b);

    /**
     * @brief calculates the sign of a number
     *
     * @tparam T
     * @param a
     * @return int
     */
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