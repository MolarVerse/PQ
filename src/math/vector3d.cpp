#include "vector3d.hpp"

using namespace vector3d;

template <class T> Vector3D<T> &Vector3D<T>::operator=(Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T> void Vector3D<T>::operator+=(const Vector3D<T> &rhs)
{
    _x += rhs[0];
    _y += rhs[1];
    _z += rhs[2];
}

template <class T> Vector3D<T> &Vector3D<T>::operator+=(const T rhs)
{
    _x += rhs;
    _y += rhs;
    _z += rhs;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<T> &rhs)
{
    _x -= rhs._x;
    _y -= rhs._y;
    _z -= rhs._z;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator-=(const T rhs)
{
    _x -= rhs;
    _y -= rhs;
    _z -= rhs;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<T> &rhs)
{
    _x *= rhs._x;
    _y *= rhs._y;
    _z *= rhs._z;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator*=(const T rhs)
{
    _x *= rhs;
    _y *= rhs;
    _z *= rhs;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator/=(const Vector3D<T> &rhs)
{
    _x /= rhs._x;
    _y /= rhs._y;
    _z /= rhs._z;
    return *this;
}

template <class T> Vector3D<T> &Vector3D<T>::operator/=(const T rhs)
{
    _x /= rhs;
    _y /= rhs;
    _z /= rhs;
    return *this;
}

template class vector3d::Vector3D<double>;
template class vector3d::Vector3D<int>;
template class vector3d::Vector3D<size_t>;