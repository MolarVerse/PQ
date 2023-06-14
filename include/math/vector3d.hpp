#ifndef _VEC3D_HPP_

#define _VEC3D_HPP_

#include <array>
#include <cstddef>
#include <iterator>
#include <cmath>
#include <iostream>

template <typename T>
using iterator = typename std::array<T, 3>::iterator;

template <typename T>
using const_iterator = typename std::array<T, 3>::const_iterator;

template <class T>
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
    explicit Vector3D(const T xyz) : _x(xyz), _y(xyz), _z(xyz){};
    Vector3D(const T x, const T y, const T z) : _x(x), _y(y), _z(z){};
    explicit Vector3D(const std::array<T, 3> &xyz) : _xyz(xyz){};
    Vector3D(const Vector3D<T> &xyz) : _xyz(xyz._xyz){};

    T &operator[](const size_t index) { return _xyz[index]; };
    const T &operator[](const size_t index) const { return _xyz[index]; };

    Vector3D &operator=(Vector3D<T> &rhs);
    const Vector3D &operator=(const Vector3D<T> &rhs);

    bool operator==(const Vector3D<T> &rhs) const { return _x == rhs._x && _y == rhs._y && _z == rhs._z; }

    Vector3D<T> &operator+=(const Vector3D<T> &rhs);
    Vector3D<T> &operator+=(const T rhs);

    Vector3D<T> &operator-=(const Vector3D<T> &rhs);
    Vector3D<T> &operator-=(const T rhs);

    Vector3D<T> &operator*=(const Vector3D<T> &rhs);
    Vector3D<T> &operator*=(const T rhs);

    Vector3D<T> &operator/=(const Vector3D<T> &rhs);
    Vector3D<T> &operator/=(const T rhs);

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
    Vector3D<T> operator+(const Vector3D<T> &rhs) const { return Vector3D<T>(_x + rhs._x, _y + rhs._y, _z + rhs._z); }
    Vector3D<T> operator+(const T rhs) const { return Vector3D<T>(_x + rhs, _y + rhs, _z + rhs); }

    Vector3D<T> operator-(const Vector3D<T> &rhs) const { return Vector3D<T>(_x - rhs._x, _y - rhs._y, _z - rhs._z); }
    Vector3D<T> operator-(const T rhs) const { return Vector3D<T>(_x - rhs, _y - rhs, _z - rhs); }
    Vector3D<T> operator-() const { return Vector3D<T>(-_x, -_y, -_z); }

    Vector3D<T> operator*(const T rhs) const { return Vector3D<T>(_x * rhs, _y * rhs, _z * rhs); }
    Vector3D<T> operator*(const Vector3D<T> &rhs) const { return Vector3D<T>(_x * rhs._x, _y * rhs._y, _z * rhs._z); }
    friend Vector3D<T> operator*(const T lhs, const Vector3D<T> &rhs) { return rhs * lhs; }

    Vector3D<T> operator/(const T rhs) const { return Vector3D<T>(_x / rhs, _y / rhs, _z / rhs); }
    Vector3D<T> operator/(const Vector3D<T> &rhs) const { return Vector3D<T>(_x / rhs._x, _y / rhs._y, _z / rhs._z); }

    constexpr iterator<T> begin() noexcept { return _xyz.begin(); };
    constexpr const_iterator<T> begin() const noexcept { return _xyz.begin(); };
    constexpr iterator<T> end() noexcept { return _xyz.end(); };
    constexpr const_iterator<T> end() const noexcept { return _xyz.end(); };

    // for static_cast
    template <class U>
    explicit operator Vector3D<U>() { return Vector3D<U>(static_cast<U>(_x), static_cast<U>(_y), static_cast<U>(_z)); };
    template <class U>
    explicit operator Vector3D<U>() const { return Vector3D<U>(static_cast<U>(_x), static_cast<U>(_y), static_cast<U>(_z)); };

    friend Vector3D<T> operator/(const T rhs, const Vector3D<T> &lhs) { return Vector3D<T>(rhs / lhs._x, rhs / lhs._y, rhs / lhs._z); }
    friend Vector3D<T> round(Vector3D<T> v) { return Vector3D<T>(round(v._x), round(v._y), round(v._z)); };
    friend Vector3D<T> ceil(Vector3D<T> v) { return Vector3D<T>(ceil(v._x), ceil(v._y), ceil(v._z)); };
    friend Vector3D<T> floor(Vector3D<T> v) { return Vector3D<T>(floor(v._x), floor(v._y), floor(v._z)); };
    friend T norm(Vector3D<T> v) { return sqrt(v._x * v._x + v._y * v._y + v._z * v._z); };
    friend T normSquared(Vector3D<T> v) { return v._x * v._x + v._y * v._y + v._z * v._z; };
    friend T minimum(Vector3D<T> v) { return std::min(v._x, std::min(v._y, v._z)); };
    friend T sum(Vector3D<T> v) { return v._x + v._y + v._z; };
    friend T prod(Vector3D<T> v) { return v._x * v._y * v._z; };
    friend T mean(Vector3D<T> v) { return sum(v) / 3; };
    friend std::ostream &operator<<(std::ostream &os, const Vector3D<T> &v) { return os << v._x << " " << v._y << " " << v._z; };
};

using Vec3D = Vector3D<double>;
using Vec3Di = Vector3D<int>;
using Vec3Dul = Vector3D<size_t>;

#endif // _VEC3D_HPP_
