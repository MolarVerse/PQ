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

#ifndef _STATIC_MATRIX_CLASS_3X3_HPP_

#define _STATIC_MATRIX_CLASS_3X3_HPP_

#include "vector3d.hpp"

namespace linearAlgebra
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
       private:
        Vector3D<Vector3D<T>> _data;

       public:
        StaticMatrix3x3() = default;

        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &data);
        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &&data);
        explicit StaticMatrix3x3(
            const Vector3D<T> &,
            const Vector3D<T> &,
            const Vector3D<T> &
        );

        StaticMatrix3x3(const T t);
        StaticMatrix3x3(const std::vector<T> &vector);

        Vector3D<T>       &operator[](const size_t index);
        const Vector3D<T> &operator[](const size_t index) const;

        friend bool operator==(
            const StaticMatrix3x3 &,
            const StaticMatrix3x3 &
        ) = default;

        StaticMatrix3x3 operator-();
        std::vector<T>  toStdVector() const;
    };
}   // namespace linearAlgebra

#include "staticMatrix3x3Class.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _STATIC_MATRIX_CLASS_3X3_HPP_