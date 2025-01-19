#ifndef __ATOMIC_OPERATIONS_INL__
#define __ATOMIC_OPERATIONS_INL__

namespace linearAlgebra
{
    /**
     * @brief atomic add operation
     *
     * @param result
     * @param value
     */
    // clang-format off
    #pragma omp declare target
    template <typename T>
    static void inline atomicAdd(T *const result, const T value)
    {
        #pragma omp atomic
        *result += value;
    }
    #pragma omp end declare target
    // clang-format on

    /**
     * @brief atomic subtract operation
     *
     * @param result
     * @param value
     */
    // clang-format off
    #pragma omp declare target
    template <typename T>
    static void inline atomicSubtract(T *const result, const T value)
    {
        #pragma omp atomic
        *result -= value;
    }
    #pragma omp end declare target

}   // namespace linearAlgebra

#endif   // __ATOMIC_OPERATIONS_INL__