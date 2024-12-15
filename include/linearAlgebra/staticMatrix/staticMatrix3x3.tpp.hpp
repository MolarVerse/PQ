/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _STATIC_MATRIX_3X3_TPP_

#define _STATIC_MATRIX_3X3_TPP_

#include "staticMatrix3x3.hpp"

#include "staticMatrix3x3Class.hpp"   // for StaticMatrix3x3
#include "vector3d.hpp"               // for Vector3D

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

    /****************************
     *                          *
     * operator+ and operator+= *
     *                          *
     ****************************/

    /**
     * @brief operator+ for two StaticMatrix3x3's
     *
     * @param StaticMatrix3x3<T> lhs, StaticMatrix3x3<T> rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator+(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<T> &rhs
    )
    {
        StaticMatrix3x3<T> result(lhs);

        result += rhs;

        return result;
    }

    /**
     * @brief operator+ for two StaticMatrix3x3's
     *
     * @param StaticMatrix3x3<T> lhs, StaticMatrix3x3<T> rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator+(const StaticMatrix3x3<T> &lhs, const T &rhs)
    {
        StaticMatrix3x3<T> result(lhs);

        result = lhs + StaticMatrix3x3<T>(rhs);

        return result;
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

    /****************************
     *                          *
     * operator- and operator-= *
     *                          *
     ****************************/

    /**
     * @brief operator- for two StaticMatrix3x3's
     *
     * @param StaticMatrix3x3<T> lhs, StaticMatrix3x3<T> rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator-(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<T> &rhs
    )
    {
        StaticMatrix3x3<T> result(lhs);

        result -= rhs;

        return result;
    }

    /**
     * @brief operator-= for two StaticMatrix3x3's
     *
     * @param lhs
     * @param rhs
     */
    template <typename T>
    void operator-=(StaticMatrix3x3<T> &lhs, const StaticMatrix3x3<T> &rhs)
    {
        lhs[0] -= rhs[0];
        lhs[1] -= rhs[1];
        lhs[2] -= rhs[2];
    }

    /****************************
     *                          *
     * operator* and operator*= *
     *                          *
     ****************************/

    /**
     * @brief operator* for two StaticMatrix3x3's
     *
     * @param StaticMatrix3x3<T> lhs, StaticMatrix3x3<T> rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator*(
        const StaticMatrix3x3<T> &lhs,
        const StaticMatrix3x3<T> &rhs
    )
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
     * @brief operator* for StaticMatrix3x3 and Vector3D
     *
     * @tparam T
     * @param mat
     * @param vec
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    Vector3D<T> operator*(const StaticMatrix3x3<T> &mat, const Vector3D<T> &vec)
    {
        Vector3D<T> result;

        result[0] = sum(mat[0] * vec);
        result[1] = sum(mat[1] * vec);
        result[2] = sum(mat[2] * vec);

        return result;
    }

    /**
     * @brief operator*= for a StaticMatrix3x3 and a scalar
     *
     * @param lhs
     * @param rhs
     */
    template <typename T>
    void operator*=(StaticMatrix3x3<T> &lhs, const T t)
    {
        lhs[0] *= t;
        lhs[1] *= t;
        lhs[2] *= t;
    }

    /****************************
     *                          *
     * operator/ and operator/= *
     *                          *
     ****************************/

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
     * @brief operator/ for scalar and StaticMatrix3x3
     *
     * @param StaticMatrix3x3<T> mat, T t
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> operator/(const T t, const StaticMatrix3x3<T> &mat)
    {
        return StaticMatrix3x3<T>(t / mat[0], t / mat[1], t / mat[2]);
    }

    /**
     * @brief operator/= for a StaticMatrix3x3 and a scalar
     *
     * @param lhs
     * @param rhs
     */
    template <typename T>
    void operator/=(StaticMatrix3x3<T> &lhs, const T t)
    {
        lhs[0] /= t;
        lhs[1] /= t;
        lhs[2] /= t;
    }

    /****************************
     *                          *
     * general matrix functions *
     *                          *
     ****************************/

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
            for (size_t j = 0; j < 3; ++j) result[i][j] = mat[j][i];

        return result;
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
     * @brief tensor product of two Vector3D's
     *
     * @details it performs v1 * v2^T
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return StaticMatrix3x3<T>
     */
    template <typename T>
    StaticMatrix3x3<T> tensorProduct(
        const Vector3D<T> &lhs,
        const Vector3D<T> &rhs
    )
    {
        StaticMatrix3x3<T> lhsMatrix{};
        StaticMatrix3x3<T> rhsMatrix{};

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
        result[0][1] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
        result[0][2] = mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0];
        result[1][0] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
        result[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
        result[1][2] = mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1];
        result[2][0] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];
        result[2][1] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];
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

    /**
     * @brief diagonal of a StaticMatrix3x3
     *
     * @param mat
     * @return Vector3D<T>
     */
    template <typename T>
    Vector3D<T> diagonal(const StaticMatrix3x3<T> &mat)
    {
        return Vector3D<T>(mat[0][0], mat[1][1], mat[2][2]);
    }

    /**
     * @brief build diagonalMatrix from a Vector3D
     *
     * @param vec
     */
    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const Vector3D<T> &vec)
    {
        StaticMatrix3x3<T> result{T()};

        result[0][0] = vec[0];
        result[1][1] = vec[1];
        result[2][2] = vec[2];

        return result;
    }

    /**
     * @brief build diagonalMatrix from a scalar
     *
     * @param t
     */
    template <typename T>
    StaticMatrix3x3<T> diagonalMatrix(const T t)
    {
        StaticMatrix3x3<T> result{T()};

        result[0][0] = t;
        result[1][1] = t;
        result[2][2] = t;

        return result;
    }

    /**
     * @brief trace of a StaticMatrix3x3
     *
     * @param mat
     * @return T
     */
    template <typename T>
    T trace(const StaticMatrix3x3<T> &mat)
    {
        return mat[0][0] + mat[1][1] + mat[2][2];
    }

    /**
     * @brief Kronecker delta
     */
    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> kroneckerDeltaMatrix()
    {
        return StaticMatrix3x3<T>(
            Vec3D{T(1), 0.0, 0.0},
            Vec3D{0.0, T(1), 0.0},
            Vec3D{0.0, 0.0, T(1)}
        );
    }

    /**
     * @brief exponential of a StaticMatrix3x3
     *
     */
    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> exp(const StaticMatrix3x3<T> &mat)
    {
        auto result = StaticMatrix3x3<T>(0.0);

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j) result[i][j] = ::exp(mat[i][j]);

        return result;
    }

    /**
     * @brief Pade approximation of the exponential of a StaticMatrix3x3
     *
     * @link https://en.wikipedia.org/wiki/Matrix_exponential
     * @link https://en.wikipedia.org/wiki/Pad%C3%A9_table
     * @link https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
     * @link
     * https://github.com/bussilab/crescale/blob/master/simplemd_anisotropic/simplemd.cpp#L351
     */
    template <typename T>
    [[nodiscard]] StaticMatrix3x3<T> expPade(const StaticMatrix3x3<T> &mat)
    {
        auto result = StaticMatrix3x3<T>(0.0);

        auto mat2 = mat * mat;
        auto mat3 = mat2 * mat;

        auto den = diagonalMatrix(1.0) - 0.5 * mat + 0.1 * mat2 - mat3 / 120.0;
        auto num = diagonalMatrix(1.0) + 0.5 * mat + 0.1 * mat2 + mat3 / 120.0;

        result = inverse(den) * num;

        return result;
    }
}   // namespace linearAlgebra

#endif   // _STATIC_MATRIX_3X3_TPP_