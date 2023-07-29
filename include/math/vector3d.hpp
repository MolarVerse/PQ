#ifndef _VEC3D_HPP_

#define _VEC3D_HPP_

#include <array>
#include <bits/stdc++.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>

namespace vector3d
{
    template <typename T> using iterator       = typename std::array<T, 3>::iterator;
    template <typename T> using const_iterator = typename std::array<T, 3>::const_iterator;

    template <class T> class Vector3D;

    using Vec3D   = Vector3D<double>;
    using Vec3Di  = Vector3D<int>;
    using Vec3Dul = Vector3D<size_t>;
}   // namespace vector3d

/**
 * @brief Vector3D class
 *
 * @note this class is a template class for all xyz objects
 *
 * @tparam T
 */
template <class T> class vector3d::Vector3D
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
    Vector3D(const Vector3D<T> &xyz) : _xyz(xyz._xyz){};
    explicit Vector3D(const T xyz) : _x(xyz), _y(xyz), _z(xyz){};
    explicit Vector3D(const std::array<T, 3> &xyz) : _xyz(xyz){};

    using value_type = T;

    /**
     * @brief index operator
     *
     * @param const size_t index
     * @return T&
     */
    T &operator[](const size_t index) { return _xyz[index]; }

    /**
     * @brief const index operator
     *
     * @param const size_t index
     * @return const T&
     */
    const T &operator[](const size_t index) const { return _xyz[index]; }

    /**
     * @brief Construct a new Vector 3D object
     *
     * @param Vector3D<T>&
     * @return Vector3D&
     */
    Vector3D &operator=(Vector3D<T> &);

    /**
     * @brief Construct a new Vector 3D object
     *
     * @param const Vector3D<T>&
     * @return Vector3D&
     */
    Vector3D &operator=(const Vector3D<T> &);

    /**
     * @brief operator ==
     *
     * @param const Vector3D<T> rhs
     * @return bool
     */
    bool operator==(const Vector3D<T> &rhs) const { return _x == rhs[0] && _y == rhs[1] && _z == rhs[2]; }

    /**
     * @brief += operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @note void to be a lot faster than with return this*
     */
    void operator+=(const Vector3D<T> &);

    /**
     * @brief += operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D&
     */
    Vector3D &operator+=(const T);

    /**
     * @brief -= operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D&
     */
    Vector3D &operator-=(const Vector3D<T> &);

    /**
     * @brief -= operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D&
     */
    Vector3D &operator-=(const T);

    /**
     * @brief *= operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D&
     */
    Vector3D &operator*=(const Vector3D<T> &);

    /**
     * @brief *= operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D&
     */
    Vector3D &operator*=(const T);

    /**
     * @brief /= operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D&
     */
    Vector3D &operator/=(const Vector3D<T> &);

    /**
     * @brief /= operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D&
     */
    Vector3D &operator/=(const T);

    /**
     * @brief + operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator+(const Vector3D<T> &rhs) const { return Vector3D<T>(_x + rhs._x, _y + rhs._y, _z + rhs._z); }

    /**
     * @brief + operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D
     */
    Vector3D<T> operator+(const T rhs) const { return Vector3D<T>(_x + rhs, _y + rhs, _z + rhs); }

    /**
     * @brief unary - operator for Vector3d
     *
     * @return Vector3D<T>
     */
    Vector3D operator-() const { return Vector3D<T>(-_x, -_y, -_z); }

    /**
     * @brief - operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator-(const Vector3D<T> &rhs) const { return {_x - rhs._x, _y - rhs._y, _z - rhs._z}; }

    /**
     * @brief - operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D
     */
    Vector3D<T> operator-(const T rhs) const { return Vector3D<T>(_x - rhs, _y - rhs, _z - rhs); }

    /**
     * @brief * operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator*(const T rhs) const { return Vector3D<T>(_x * rhs, _y * rhs, _z * rhs); }

    /**
     * @brief * operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator*(const Vector3D<T> &rhs) const { return {_x * rhs._x, _y * rhs._y, _z * rhs._z}; }

    /**
     * @brief * operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D
     */
    friend Vector3D<T> operator*(const T lhs, const Vector3D<T> &rhs) { return rhs * lhs; }

    /**
     * @brief / operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator/(const T rhs) const { return Vector3D<T>(_x / rhs, _y / rhs, _z / rhs); }

    /**
     * @brief / operator for two Vector3d objects
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    Vector3D<T> operator/(const Vector3D<T> &rhs) const { return Vector3D<T>(_x / rhs._x, _y / rhs._y, _z / rhs._z); }

    /**
     * @brief / operator for a Vector3d object and a scalar
     *
     * @param const T
     * @return Vector3D
     */
    friend Vector3D<T> operator/(const T rhs, const Vector3D<T> &lhs)
    {
        return Vector3D<T>(rhs / lhs._x, rhs / lhs._y, rhs / lhs._z);
    }

    /**
     * @brief < operator for vector3d and scalar
     *
     * @param const T t
     * @return bool
     *
     * @note returns true if all members of vector are less than t
     */
    bool operator<(const T t) const { return _x < t && _y < t && _z < t; }

    /**
     * @brief > operator for vector3d and scalar
     *
     * @param const T t
     * @return bool
     *
     * @note returns true if all members of vector are greater than t
     */
    bool operator>(const T t) const { return _x > t && _y > t && _z > t; }

    /**
     * @brief fabs of all entries of vector
     *
     * @param const Vector3D<T>&
     * @return Vector3D
     */
    friend Vector3D fabs(const Vector3D &v) { return Vector3D<T>(fabs(v._x), fabs(v._y), fabs(v._z)); }

    /**
     * @brief begin iterator for vector3d
     *
     * @return constexpr const_iterator<T>
     */
    constexpr const_iterator<T> begin() const noexcept { return _xyz.begin(); }

    /**
     * @brief end iterator for vector3d
     *
     * @return constexpr const_iterator<T>
     */
    constexpr const_iterator<T> end() const noexcept { return _xyz.end(); }

    /**
     * @brief static cast of all vector members
     *
     * @tparam U
     * @return Vector3D<U>
     */
    template <class U> explicit operator Vector3D<U>() const
    {
        return Vector3D<U>(static_cast<U>(_x), static_cast<U>(_y), static_cast<U>(_z));
    }

    /**
     * @brief round all entries of vector
     *
     * @param v
     * @return Vector3D<T>
     */
    friend Vector3D<T> round(const Vector3D<T> &v) { return {rint(v[0]), rint(v[1]), rint(v[2])}; }

    /**
     * @brief ceil all entries of vector
     *
     * @param v
     * @return Vector3D<T>
     */
    friend Vector3D<T> ceil(Vector3D<T> v) { return Vector3D<T>(ceil(v._x), ceil(v._y), ceil(v._z)); }

    /**
     * @brief floor all entries of vector
     *
     * @param v
     * @return Vector3D<T>
     */
    friend Vector3D<T> floor(Vector3D<T> v) { return Vector3D<T>(floor(v._x), floor(v._y), floor(v._z)); }

    /**
     * @brief norm of vector
     *
     * @param v
     * @return T
     */
    friend T norm(Vector3D<T> v) { return sqrt(v._x * v._x + v._y * v._y + v._z * v._z); }

    /**
     * @brief norm squared of vector
     *
     * @param v
     * @return T
     */
    friend T normSquared(Vector3D<T> v) { return v._x * v._x + v._y * v._y + v._z * v._z; }

    /**
     * @brief minimum of vector
     *
     * @param v
     * @return T
     */
    friend T minimum(Vector3D<T> v) { return std::min(v._x, std::min(v._y, v._z)); }

    /**
     * @brief sum of vector
     *
     * @param v
     * @return T
     */
    friend T sum(Vector3D<T> v) { return v._x + v._y + v._z; }

    /**
     * @brief product of vector
     *
     * @param v
     * @return T
     */
    friend T prod(Vector3D<T> v) { return v._x * v._y * v._z; }

    /**
     * @brief mean of vector
     *
     * @param v
     * @return T
     */
    friend T mean(Vector3D<T> v) { return sum(v) / 3; }

    /**
     * @brief scalar_product of two vectors
     *
     * @param v1
     * @param v2
     */
    friend T dot(Vector3D<T> v1, Vector3D<T> v2) { return v1._x * v2._x + v1._y * v2._y + v1._z * v2._z; }

    /**
     * @brief ostream operator for vector3d
     *
     * @param os
     * @param v
     * @return std::ostream&
     */
    friend std::ostream &operator<<(std::ostream &os, const Vector3D<T> &v) { return os << v._x << " " << v._y << " " << v._z; }
};

#endif   // _VEC3D_HPP_
