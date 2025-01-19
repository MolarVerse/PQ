#ifndef __TENSOR_PRODUCT_INL__
#define __TENSOR_PRODUCT_INL__

namespace linearAlgebra
{
    /**
     * @brief tensor product of two vectors
     *
     * @T* const dst
     * @const T  a1
     * @const T  a2
     * @const T  a3
     * @const T  b1
     * @const T  b2
     * @const T  b3
     */
    template <typename T>
    void static inline tensorProduct(
        T* const dst,
        const T  a1,
        const T  a2,
        const T  a3,
        const T  b1,
        const T  b2,
        const T  b3
    )
    {
        dst[0] = a1 * b1;
        dst[1] = a1 * b2;
        dst[2] = a1 * b3;
        dst[3] = a2 * b1;
        dst[4] = a2 * b2;
        dst[5] = a2 * b3;
        dst[6] = a3 * b1;
        dst[7] = a3 * b2;
        dst[8] = a3 * b3;
    }

    /**
     * @brief add tensor product of two vectors to a matrix
     *
     * @T* const dst
     * @const T  a1
     * @const T  a2
     * @const T  a3
     * @const T  b1
     * @const T  b2
     * @const T  b3
     */
    template <typename T>
    static void inline addTensorProduct(
        T* const dst,
        const T  a1,
        const T  a2,
        const T  a3,
        const T  b1,
        const T  b2,
        const T  b3
    )
    {
        dst[0] += a1 * b1;
        dst[1] += a1 * b2;
        dst[2] += a1 * b3;
        dst[3] += a2 * b1;
        dst[4] += a2 * b2;
        dst[5] += a2 * b3;
        dst[6] += a3 * b1;
        dst[7] += a3 * b2;
        dst[8] += a3 * b3;
    }

}   // namespace linearAlgebra

#endif   // __TENSOR_PRODUCT_INL__