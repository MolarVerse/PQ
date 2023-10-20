/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &data) : _data(data) {}
        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &&data) : _data(std::move(data)) {}

        explicit StaticMatrix3x3(const Vector3D<T> &row1, const Vector3D<T> &row2, const Vector3D<T> &row3)
            : _data(row1, row2, row3){};

        StaticMatrix3x3(const T t)
        {
            _data[0] = Vector3D<T>(t);
            _data[1] = Vector3D<T>(t);
            _data[2] = Vector3D<T>(t);
        }

        StaticMatrix3x3(const std::vector<T> &vector)
        {
            if (vector.size() != 9)
            {
                throw std::runtime_error("vector size must be 9");
            }

            _data[0] = Vector3D<T>(vector[0], vector[1], vector[2]);
            _data[1] = Vector3D<T>(vector[3], vector[4], vector[5]);
            _data[2] = Vector3D<T>(vector[6], vector[7], vector[8]);
        }

        /**
         * @brief index operator
         *
         * @param const size_t index
         * @return std::vector<T> &
         */
        Vector3D<T> &operator[](const size_t index) { return _data[index]; }

        /**
         * @brief index operator
         *
         * @param const size_t index
         * @return const std::vector<T> &
         */
        const Vector3D<T> &operator[](const size_t index) const { return _data[index]; }

        /**
         * @brief operator== for two StaticMatrix3x3's
         *
         * @param rhs
         * @return bool
         */
        friend bool operator==(const StaticMatrix3x3 &lhs, const StaticMatrix3x3 &rhs) = default;

        /**
         * @brief unary operator- for StaticMatrix3x3
         *
         * @return StaticMatrix3x3
         */
        StaticMatrix3x3 operator-() { return StaticMatrix3x3(-_data[0], -_data[1], -_data[2]); }

        /**
         * @brief operator+ for two StaticMatrix3x3's
         *
         * @param rhs
         * @return StaticMatrix3x3
         */
        std::vector<T> toStdVector() const
        {
            std::vector<T> result;
            result.reserve(9);
            for (const auto &row : _data)
            {
                for (const auto &element : row)
                {
                    result.push_back(element);
                }
            }
            return result;
        }
    };
}   // namespace linearAlgebra

#endif   // _STATIC_MATRIX_CLASS_3X3_HPP_