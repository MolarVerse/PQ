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

#include "vector3d.hpp"

using namespace linearAlgebra;

template <class T>
Vector3D<T> &Vector3D<T>::operator=(Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T>
void Vector3D<T>::operator+=(const Vector3D<T> &rhs)
{
    _x += rhs._x;
    _y += rhs._y;
    _z += rhs._z;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator+=(const T rhs)
{
    _x += rhs;
    _y += rhs;
    _z += rhs;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<T> &rhs)
{
    _x -= rhs._x;
    _y -= rhs._y;
    _z -= rhs._z;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator-=(const T rhs)
{
    _x -= rhs;
    _y -= rhs;
    _z -= rhs;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<T> &rhs)
{
    _x *= rhs._x;
    _y *= rhs._y;
    _z *= rhs._z;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator*=(const T rhs)
{
    _x *= rhs;
    _y *= rhs;
    _z *= rhs;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator/=(const Vector3D<T> &rhs)
{
    _x /= rhs._x;
    _y /= rhs._y;
    _z /= rhs._z;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator/=(const T rhs)
{
    _x /= rhs;
    _y /= rhs;
    _z /= rhs;
    return *this;
}

template class linearAlgebra::Vector3D<double>;
template class linearAlgebra::Vector3D<int>;
template class linearAlgebra::Vector3D<size_t>;
template class linearAlgebra::Vector3D<Vector3D<double>>;