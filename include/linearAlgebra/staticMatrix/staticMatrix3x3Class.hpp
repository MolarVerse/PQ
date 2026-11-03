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

#ifndef LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_CLASS_HPP
#define LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_CLASS_HPP

#include "vector3d.hpp"

namespace linalg
{
    template <typename T>
    class StaticMatrix3x3;

    using tensor3D = StaticMatrix3x3<double>;

    /**
     * @class StaticMatrix3x3
     *
     * @brief template Matrix class with 3 rows and 3 columns
     *
     * @tparam T
     */
    template <typename T>
    class StaticMatrix3x3
    {
       public:
        static constexpr auto size = 3;

       private:
        Vector3D<Vector3D<T>> _data;

        constexpr static auto _nElements = 9;

       public:
        StaticMatrix3x3() = default;

        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &data);
        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &&data);
        explicit StaticMatrix3x3(
            const Vector3D<T> &row1,
            const Vector3D<T> &row2,
            const Vector3D<T> &row3
        );

        // NOLINTBEGIN(google-explicit-constructor, hicpp-explicit-conversions)
        StaticMatrix3x3(const T &mat);
        // NOLINTEND(google-explicit-constructor, hicpp-explicit-conversions)
        explicit StaticMatrix3x3(const std::vector<T> &vector);

        Vector3D<T>       &operator[](size_t index);
        const Vector3D<T> &operator[](size_t index) const;

        friend bool operator==(
            const StaticMatrix3x3 &lhs,
            const StaticMatrix3x3 &rhs
        ) = default;

        StaticMatrix3x3 operator-();
        std::vector<T>  toStdVector() const;

        void setDiagonal(const T &value);
        void setDiagonal(const Vector3D<T> &values);
    };
}   // namespace linalg

#ifndef LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_CLASS_TPP
#include "staticMatrix3x3Class.tpp"   // IWYU pragma: export
#endif

#endif   // LINEAR_ALGEBRA__STATIC_MATRIX__STATIC_MATRIX3X3_CLASS_HPP
