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

#ifndef __VECTOR3D_CLASS_TPP__

#define __VECTOR3D_CLASS_TPP__

#include "concepts/vector3dConcepts.hpp"
#include "vector3dClass.hpp"

namespace linearAlgebra
{
    /********************
     *                  *
     * assign operators *
     *                  *
     ********************/

    /***************
     * = operators *
     ***************/

    /**
     * @brief move assignment operator
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator=(Vector3D<T> &rhs)
    {
        _x = rhs._x;
        _y = rhs._y;
        _z = rhs._z;
        return *this;
    }

    /**
     * @brief copy assignment operator
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &rhs)
    {
        _x = rhs._x;
        _y = rhs._y;
        _z = rhs._z;
        return *this;
    }

    /****************
     * += operators *
     ****************/

    /**
     * @brief operator += inplace
     *
     * @param const Vector3D<T> &rhs
     */
    template <class T>
    void Vector3D<T>::operator+=(const Vector3D<T> &rhs)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        _x += rhs._x;
        _y += rhs._y;
        _z += rhs._z;
    }

    /**
     * @brief += operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator+=(const T rhs)
    requires pq::Arithmetic<T>
    {
        _x += rhs;
        _y += rhs;
        _z += rhs;
        return *this;
    }

    /****************
     * -= operators *
     ****************/

    /**
     * @brief operator -=
     *
     * @param const Vector3D<T> &rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<T> &rhs)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        _x -= rhs._x;
        _y -= rhs._y;
        _z -= rhs._z;
        return *this;
    }

    /**
     * @brief operator -=
     *
     * @param const T rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator-=(const T rhs)
    requires pq::Arithmetic<T>
    {
        _x -= rhs;
        _y -= rhs;
        _z -= rhs;
        return *this;
    }

    /****************
     * *= operators *
     ****************/

    /**
     * @brief operator *=
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<T> &rhs)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        _x *= rhs._x;
        _y *= rhs._y;
        _z *= rhs._z;
        return *this;
    }

    /**
     * @brief operator *=
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator*=(const T rhs)
    requires pq::Arithmetic<T>
    {
        _x *= rhs;
        _y *= rhs;
        _z *= rhs;
        return *this;
    }

    /****************
     * /= operators *
     ****************/

    /**
     * @brief operator /=
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator/=(const Vector3D<T> &rhs)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        _x /= rhs._x;
        _y /= rhs._y;
        _z /= rhs._z;
        return *this;
    }

    /**
     * @brief operator /=
     *
     * @tparam T
     * @param rhs
     * @return Vector3D<T>&
     */
    template <class T>
    Vector3D<T> &Vector3D<T>::operator/=(const T rhs)
    requires pq::Arithmetic<T>
    {
        _x /= rhs;
        _y /= rhs;
        _z /= rhs;
        return *this;
    }

    /**********************
     *                    *
     * indexing operators *
     *                    *
     **********************/

    /**
     * @brief index operator
     *
     * @param const size_t index
     * @return T&
     */
    template <class T>
    T &Vector3D<T>::operator[](const size_t index)
    {
        return _xyz[index];
    }

    /**
     * @brief const index operator
     *
     * @param const size_t index
     * @return const T&
     */
    template <class T>
    const T &Vector3D<T>::operator[](const size_t index) const
    {
        return _xyz[index];
    }

    /*******************
     *                 *
     * unary operators *
     *                 *
     *******************/

    /**
     * @brief unary - operator for Vector3d
     *
     * @return Vector3D<T>
     */
    template <class T>
    Vector3D<T> Vector3D<T>::operator-() const
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        return Vector3D<T>(-_x, -_y, -_z);
    }

    /********************
     *                  *
     * iterator methods *
     *                  *
     ********************/

    /**
     * @brief begin iterator
     *
     * @return constexpr std::array<T, 3>::iterator
     */
    template <class T>
    constexpr std::array<T, 3>::const_iterator Vector3D<T>::begin(
    ) const noexcept
    {
        return _xyz.begin();
    }

    /**
     * @brief end iterator
     *
     * @return constexpr std::array<T, 3>::iterator
     */
    template <class T>
    constexpr std::array<T, 3>::const_iterator Vector3D<T>::end() const noexcept
    {
        return _xyz.end();
    }

    /*******************
     * casting methods *
     *******************/

    /**
     * @brief static cast of all vector members
     *
     * @tparam U
     * @return Vector3D<U>
     */
    template <class T>
    template <class U>
    Vector3D<T>::operator Vector3D<U>() const
    {
        return Vector3D<U>(
            static_cast<U>(_x),
            static_cast<U>(_y),
            static_cast<U>(_z)
        );
    }

    /**
     * @brief cast to std::vector
     *
     * @return std::vector<T>
     */
    template <class T>
    std::vector<T> Vector3D<T>::toStdVector()
    {
        return std::vector<T>(_xyz.begin(), _xyz.end());
    }

}   // namespace linearAlgebra

#endif   // __VECTOR3D_CLASS_TPP__