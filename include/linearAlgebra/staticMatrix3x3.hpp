#ifndef _STATIC_MATRIX_3X3_HPP_

#define _STATIC_MATRIX_3X3_HPP_

#include "staticMatrix3x3Class.hpp"
#include "vector3d.hpp"

namespace linearAlgebra
{

    /**
     * @brief ostream operator for vector3d
     *
     * @param os
     * @param v
     * @return std::ostream&
     */
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const StaticMatrix3x3<T> &mat)
    {
        return os << "[[" << mat[0] << "]\n"
                  << " [" << mat[1] << "]\n"
                  << " [" << mat[2] << "]]";
    }

    /**
     * @brief operator+= for two StaticMatrix3x3's
     *
     * @param lhs
     * @param rhs
     */
    template <typename T>
    void operator+=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs)
    {
        lhs[0] += rhs[0];
        lhs[1] += rhs[1];
        lhs[2] += rhs[2];
    }

    /**
     * @brief transpose a StaticMatrix3x3
     *
     * @param mat
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> transpose(const StaticMatrix3x3<T> &mat)
    {
        StaticMatrix3x3<T> result;

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                result[i][j] = mat[j][i];

        return result;
    }

    /**
     * @brief operator* for two StaticMatrix3x3's
     *
     * @param StaticMatrix3x3<T> lhs, StaticMatrix3x3<T> rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator*(const StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs)
    {
        StaticMatrix3x3<T> result;
        StaticMatrix3x3<T> rhsTransposed = transpose(rhs);

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                result[i][j] = sum(lhs[i] * rhsTransposed[j]);

        return result;
    }

    /**
     * @brief operator* for StaticMatrix3x3 and scalar
     *
     * @param StaticMatrix3x3<T> mat, T t
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator*(const StaticMatrix3x3<T> &mat, const T t)
    {
        return StaticMatrix3x3<T>(mat[0] * t, mat[1] * t, mat[2] * t);
    }

    /**
     * @brief operator* for StaticMatrix3x3 and scalar
     *
     * @param T t, StaticMatrix3x3<T> mat
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator*(const T t, const StaticMatrix3x3<T> &mat)
    {
        return StaticMatrix3x3<T>(mat[0] * t, mat[1] * t, mat[2] * t);
    }

    /**
     * @brief operator/ for StaticMatrix3x3 and scalar
     *
     * @param StaticMatrix3x3<T> mat, T t
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator/(const StaticMatrix3x3<T> &mat, const T t)
    {
        return StaticMatrix3x3<T>(mat[0] / t, mat[1] / t, mat[2] / t);
    }

    /**
     * @brief determinant of a StaticMatrix3x3
     *
     * @param mat
     * @return T
     */
    template <typename T>
    T det(const StaticMatrix3x3<T> &mat)
    {
        auto result = mat[0][0] * mat[1][1] * mat[2][2];

        result += mat[0][1] * mat[1][2] * mat[2][0];
        result += mat[0][2] * mat[1][0] * mat[2][1];
        result -= mat[0][2] * mat[1][1] * mat[2][0];
        result -= mat[0][1] * mat[1][0] * mat[2][2];
        result -= mat[0][0] * mat[1][2] * mat[2][1];

        return result;
    }

    /**
     * @brief vector product of two Vector3D's
     *
     * @details it performs v1 * v2^T
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> vectorProduct(const Vector3D<T> &lhs, const Vector3D<T> &rhs)
    {

        StaticMatrix3x3<T> lhsMatrix;
        StaticMatrix3x3<T> rhsMatrix;

        lhsMatrix[0] = lhs;
        rhsMatrix[0] = rhs;

        lhsMatrix = transpose(lhsMatrix);

        return lhsMatrix * rhsMatrix;
    }

    /**
     * @brief cofactor matrix of a StaticMatrix3x3
     *
     * @param mat
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> cofactorMatrix(const StaticMatrix3x3<T> &mat)
    {
        StaticMatrix3x3<T> result;

        result[0][0] = mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1];
        result[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
        result[0][2] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];
        result[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
        result[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
        result[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];
        result[2][0] = mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0];
        result[2][1] = mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1];
        result[2][2] = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

        return result;
    }

    /**
     * @brief inverse of a StaticMatrix3x3
     *
     * @param mat
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> inverse(const StaticMatrix3x3<T> &mat)
    {
        StaticMatrix3x3<T> result = cofactorMatrix(mat);

        result = transpose(result);

        return result / det(mat);
    }

}   // namespace linearAlgebra

#endif   // _STATIC_MATRIX_3X3_HPP_