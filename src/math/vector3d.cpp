#include "vector3d.hpp"

template <class T>
Vector3D<T> &Vector3D<T>::operator=(Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T>
const Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &rhs)
{
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    return *this;
}

template <class T>
Vector3D<T> &Vector3D<T>::operator+=(const Vector3D<T> &rhs)
{
    _x += rhs._x;
    _y += rhs._y;
    _z += rhs._z;
    return *this;
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

template class Vector3D<double>;
template class Vector3D<int>;
template class Vector3D<size_t>;