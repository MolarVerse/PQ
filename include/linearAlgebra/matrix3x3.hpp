#ifndef _MATRIX_3X3_HPP_

#define _MATRIX_3X3_HPP_

#include "matrixnx3.hpp"
#include "vector3d.hpp"

namespace linearAlgebra
{

    /**
     * @class Matrix3x3
     *
     * @brief template Matrix class with 3 rows and 3 columns
     *
     * @tparam T
     */
    template <typename T>
    class Matrix3x3 : public MatrixNX3<T>
    {
      public:
        Matrix3x3() : MatrixNX3<T>(3) {}
    };

    // Matrix3x3<T> vectorProduct(const Vector3D<T> &lhs, const Vector3D<T> &rhs)
    // {
    //     Matrix3x3<T> lhs;
    //     Matrix3x3<T> rhs;

    //     return result;
    // }

}   // namespace linearAlgebra

#endif   // _MATRIX_3X3_HPP_