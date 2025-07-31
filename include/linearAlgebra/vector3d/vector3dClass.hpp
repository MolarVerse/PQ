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

#ifndef _VECTOR3D_CLASS_HPP_

#define _VECTOR3D_CLASS_HPP_

#include <array>         // for array
#include <cmath>         // for ceil, fabs, floor, rint, sqrt
#include <cstddef>       // for size_t
#include <iostream>      // for ostream
#include <type_traits>   // for is_fundamental_v
#include <vector>        // for vector

#include "concepts/vector3dConcepts.hpp"

namespace linearAlgebra
{
    template <typename T>
    using iterator = typename std::array<T, 3>::iterator;

    template <typename T>
    using const_iterator = typename std::array<T, 3>::const_iterator;

    template <class T>
    class Vector3D;   // forward declaration

    using Vec3D   = Vector3D<double>;
    using Vec3Di  = Vector3D<int>;
    using Vec3Dul = Vector3D<size_t>;

    /**
     * @brief Vector3D class
     *
     * @note this class is a template class for all xyz objects
     *
     * @tparam T
     */
    template <typename T>
    class Vector3D
    {
       private:
        union
        {
            std::array<T, 3> _xyz;
            struct
            {
                T _x;
                T _y;
                T _z;
            };
        };

       public:
        ~Vector3D() = default;

        Vector3D() = default;
        Vector3D(const T x, const T y, const T z) : _x(x), _y(y), _z(z) {}
        Vector3D(const Vector3D<T> &xyz) : _xyz(xyz._xyz) {}
        Vector3D(const T xyz) : _x(xyz), _y(xyz), _z(xyz) {}

        using value_type = T;

        /********************
         * assign operators *
         ********************/

        // = operators
        Vector3D &operator=(Vector3D<T> &);
        Vector3D &operator=(const Vector3D<T> &);

        // += operators
        void operator+=(const Vector3D<T> &)
        requires(pq::ArithmeticVector3D<T> || pq::Arithmetic<T>);

        Vector3D &operator+=(const T)
        requires pq::Arithmetic<T>;

        // -= operators
        Vector3D &operator-=(const Vector3D<T> &)
        requires(pq::ArithmeticVector3D<T> || pq::Arithmetic<T>);

        Vector3D &operator-=(const T)
        requires pq::Arithmetic<T>;

        // *= operators
        Vector3D &operator*=(const Vector3D<T> &)
        requires(pq::ArithmeticVector3D<T> || pq::Arithmetic<T>);

        Vector3D &operator*=(const T)
        requires pq::Arithmetic<T>;

        // /= operators
        Vector3D &operator/=(const Vector3D<T> &)
        requires(pq::ArithmeticVector3D<T> || pq::Arithmetic<T>);

        Vector3D &operator/=(const T)
        requires pq::Arithmetic<T>;

        /**********************
         * indexing operators *
         **********************/

        T       &operator[](const size_t index);
        const T &operator[](const size_t index) const;

        /*******************
         * unary operators *
         *******************/

        Vector3D operator-() const
        requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>;

        /********************
         * iterator methods *
         ********************/

        constexpr const_iterator<T> begin() const noexcept;
        constexpr const_iterator<T> end() const noexcept;

        /*******************
         * casting methods *
         *******************/

        template <class U>
        explicit operator Vector3D<U>() const;

        std::vector<T> toStdVector();
    };

}   // namespace linearAlgebra

#include "vector3dClass.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _VECTOR3D_CLASS_HPP_
