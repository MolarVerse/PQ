#ifndef __CROSS_INL__
#define __CROSS_INL__

#include <vector>   // for vector

namespace linearAlgebra
{
    /**
     * @brief calculate the cross product of two vectors
     *
     * @param a
     * @param b
     *
     * @return cross product of a and b
     */
    template <typename T>
    std::vector<T> cross(const T* const a, std::vector<T> b)
    {
        const auto x = a[1] * b[2] - a[2] * b[1];
        const auto y = a[2] * b[0] - a[0] * b[2];
        const auto z = a[0] * b[1] - a[1] * b[0];

        return std::vector<T>({x, y, z});
    }

    /**
     * @brief calculate the cross product of two vectors
     *
     * @param a
     * @param b
     *
     * @return cross product of a and b
     */
#pragma omp declare target
    template <typename T>
    void static inline cross(
        T* const       result,
        const T* const a,
        const T* const b
    )
    {
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    }
#pragma omp end declare target

    /**
     * @brief calculate the cross product of two vectors
     *
     * @param a
     * @param b
     *
     * @return cross product of a and b
     */
#pragma omp declare target
    template <typename T>
    void static inline cross(
        T* const       result,
        const T* const a,
        const T        b1,
        const T        b2,
        const T        b3
    )
    {
        result[0] = a[1] * b3 - a[2] * b2;
        result[1] = a[2] * b1 - a[0] * b3;
        result[2] = a[0] * b2 - a[1] * b1;
    }
#pragma omp end declare target

}   // namespace linearAlgebra

#endif   // __CROSS_INL__