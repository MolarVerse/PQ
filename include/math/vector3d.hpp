#ifndef _VEC3D_HPP_

#define _VEC3D_HPP_

#include <array>
#include <cstddef>

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
    Vector3D() = default;
    Vector3D(const T x, const T y, const T z) : _x(x), _y(y), _z(z){};
    explicit Vector3D(const std::array<T, 3> &xyz) : _xyz(xyz){};

    T &operator[](const size_t index) { return _xyz[index]; };
    const T &operator[](const size_t index) const { return _xyz[index]; };

    Vector3D &operator=(const Vector3D<T> &rhs)
    {
        _x = rhs._x;
        _y = rhs._y;
        _z = rhs._z;
        return *this;
    }

    // standard getter and setters
    void setX(const T x) { _x = x; };
    [[nodiscard]] T getX() const { return _x; };

    void setY(const T y) { _y = y; };
    [[nodiscard]] T getY() const { return _y; };

    void setZ(const T z) { _z = z; };
    [[nodiscard]] T getZ() const { return _z; };

    void setData(const std::array<T, 3> &xyz) { _xyz = xyz; };
    [[nodiscard]] std::array<T, 3> getData() const { return _xyz; };

    // operator overloading
    Vector3D<T> operator+(const Vector3D<T> &rhs) const
    {
        return Vector3D<T>(_x + rhs._x, _y + rhs._y, _z + rhs._z);
    }

    Vector3D<T> operator-(const Vector3D<T> &rhs) const
    {
        return Vector3D<T>(_x - rhs._x, _y - rhs._y, _z - rhs._z);
    }

    Vector3D<T> operator*(const T rhs) const
    {
        return Vector3D<T>(_x * rhs, _y * rhs, _z * rhs);
    }

    Vector3D<T> operator/(const T rhs) const
    {
        return Vector3D<T>(_x / rhs, _y / rhs, _z / rhs);
    }

    Vector3D<T> &operator+=(const Vector3D<T> &rhs)
    {
        _x += rhs._x;
        _y += rhs._y;
        _z += rhs._z;
        return *this;
    }

    Vector3D<T> &operator-=(const Vector3D<T> &rhs)
    {
        _x -= rhs._x;
        _y -= rhs._y;
        _z -= rhs._z;
        return *this;
    }

    Vector3D<T> &operator*=(const T rhs)
    {
        _x *= rhs;
        _y *= rhs;
        _z *= rhs;
        return *this;
    }
};

using Vec3D = Vector3D<double>;
using Vec3Df = Vector3D<float>;
using Vec3Di = Vector3D<int>;

#endif // _VEC3D_HPP_
