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

#ifndef LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_CLASS_TPP
#define LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_CLASS_TPP

#include <utility>

#include "concepts/vector3dConcepts.hpp"
#include "vector3dClass.hpp"

namespace linalg
{
    /**
     * @brief default constructor, the components are value-initialised
     *
     * @details Zero for arithmetic types. A defaulted constructor left the
     * components uninitialised, which printed garbage (once inf) into the
     * first output line of a run.
     *
     * @tparam T
     */
    template <typename T>
    constexpr Vector3D<T>::Vector3D() : x(), y(), z()
    {
    }

    /**
     * @brief constructor from the three components
     *
     * @tparam T
     * @param xValue
     * @param yValue
     * @param zValue
     */
    template <typename T>
    Vector3D<T>::Vector3D(T xValue, T yValue, T zValue)
        : x(xValue), y(yValue), z(zValue)
    {
    }

    /**
     * @brief copy constructor
     *
     * @tparam T
     * @param xyz_
     */
    template <typename T>
    Vector3D<T>::Vector3D(const Vector3D<T> &xyz_) : xyz(xyz_.xyz)
    {
    }

    /**
     * @brief move constructor
     *
     * @tparam T
     * @param xyz_
     */
    template <typename T>
    Vector3D<T>::Vector3D(Vector3D<T> &&xyz_) noexcept
        : xyz(std::move(xyz_.xyz))
    {
    }

    /**
     * @brief constructor setting all three components to the same value
     *
     * @tparam T
     * @param xyz_
     */
    template <typename T>
    Vector3D<T>::Vector3D(T xyz_) : x(xyz_), y(xyz_), z(xyz_)
    {
    }

    /**
     * @brief vector with all components zero (uses T::zero() for
     * quantities, value initialization otherwise)
     *
     * @tparam T
     * @return Vector3D<T>
     */
    template <typename T>
    constexpr Vector3D<T> Vector3D<T>::zero()
    {
        if constexpr (requires { T::zero(); })
        {
            return Vector3D(T::zero(), T::zero(), T::zero());
        }
        else
        {
            return Vector3D(T{}, T{}, T{});
        }
    }

    /**
     * @brief converting constructor between quantity vectors
     *
     * @tparam T
     * @tparam U
     * @param vec
     */
    template <typename T>
    template <typename U>
    requires(!std::same_as<U, T> && mstd::is_quantity_like_v<T> &&
             mstd::is_quantity_like_v<U> && std::convertible_to<U, T>)
    Vector3D<T>::Vector3D(const Vector3D<U> &vec)
        : x(static_cast<T>(vec[0])),
          y(static_cast<T>(vec[1])),
          z(static_cast<T>(vec[2]))
    {
    }

    /**
     * @brief constructor from a quantity holding a Vector3D<double>
     *
     * @tparam T
     * @tparam Q
     * @param quantity
     */
    template <typename T>
    template <typename Q>
    requires(mstd::is_quantity_like_v<T> && mstd::is_quantity_like_v<Q> &&
             !std::same_as<Q, T> &&
             std::same_as<typename Q::rep, Vector3D<double>>)
    Vector3D<T>::Vector3D(const Q &quantity)
        : x(quantity.numerical_value_in(Q::unit)[0] * Q::unit),
          y(quantity.numerical_value_in(Q::unit)[1] * Q::unit),
          z(quantity.numerical_value_in(Q::unit)[2] * Q::unit)
    {
    }

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
     * @param vec
     * @return Vector3D<T>&
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator=(Vector3D<T> &&vec) noexcept
    {
        if (this == &vec)
            return *this;

        x = vec.x;
        y = vec.y;
        z = vec.z;
        return *this;
    }

    /**
     * @brief copy assignment operator
     *
     * @tparam T
     * @param vec
     * @return Vector3D<T>&
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &vec)
    {
        if (this == &vec)
            return *this;

        x = vec.x;
        y = vec.y;
        z = vec.z;
        return *this;
    }

    /****************
     * += operators *
     ****************/

    /**
     * @brief operator += inplace
     *
     * @param vec
     */
    template <typename T>
    void Vector3D<T>::operator+=(const Vector3D<T> &vec)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T> ||
             mstd::is_quantity_like_v<T>
    {
        x += vec.x;
        y += vec.y;
        z += vec.z;
    }

    /**
     * @brief += operator for two Vector3d objects
     *
     * @param vec
     * @return `Vector3D<T>&` (inplace addition)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator+=(const T &vec)
    requires pq::Arithmetic<T> || mstd::is_quantity_like_v<T>
    {
        x += vec;
        y += vec;
        z += vec;
        return *this;
    }

    /****************
     * -= operators *
     ****************/

    /**
     * @brief operator -=
     *
     * @param vec
     * @return `Vector3D<T>&` (inplace subtraction)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<T> &vec)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T> ||
             mstd::is_quantity_like_v<T>
    {
        x -= vec.x;
        y -= vec.y;
        z -= vec.z;
        return *this;
    }

    /**
     * @brief operator -=
     *
     * @param vec
     * @return `Vector3D<T>&` (inplace subtraction)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator-=(const T &vec)
    requires pq::Arithmetic<T> || mstd::is_quantity_like_v<T>
    {
        x -= vec;
        y -= vec;
        z -= vec;
        return *this;
    }

    /****************
     * *= operators *
     ****************/

    /**
     * @brief operator *=
     *
     * @tparam T
     * @param vec
     * @return `Vector3D<T>&` (inplace multiplication)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<T> &vec)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        x *= vec.x;
        y *= vec.y;
        z *= vec.z;
        return *this;
    }

    /**
     * @brief operator *=
     *
     * @tparam T
     * @param vec
     * @return `Vector3D<T>&` (inplace multiplication)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator*=(const T &vec)
    requires pq::Arithmetic<T>
    {
        x *= vec;
        y *= vec;
        z *= vec;
        return *this;
    }

    /****************
     * /= operators *
     ****************/

    /**
     * @brief operator /=
     *
     * @tparam T
     * @param vec
     * @return `Vector3D<T>&` (inplace division)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator/=(const Vector3D<T> &vec)
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T>
    {
        x /= vec.x;
        y /= vec.y;
        z /= vec.z;
        return *this;
    }

    /**
     * @brief operator /=
     *
     * @tparam T
     * @param vec
     * @return `Vector3D<T>&` (inplace division)
     */
    template <typename T>
    Vector3D<T> &Vector3D<T>::operator/=(const T &vec)
    requires pq::Arithmetic<T>
    {
        x /= vec;
        y /= vec;
        z /= vec;
        return *this;
    }

    /**
     * @brief operator /= for quantity elements and a dimensionless scalar
     *
     * @tparam T
     * @tparam S
     * @param scalar
     * @return `Vector3D<T>&` (inplace division)
     */
    template <typename T>
    template <typename S>
    requires(
        mstd::is_quantity_like_v<T> && !std::same_as<S, T> &&
        requires(T &elem, const S &scalar) { elem /= scalar; }
    )
    Vector3D<T> &Vector3D<T>::operator/=(const S &scalar)
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /**
     * @brief operator *= for quantity elements and a dimensionless scalar
     *
     * @tparam T
     * @tparam S
     * @param scalar
     * @return `Vector3D<T>&` (inplace multiplication)
     */
    template <typename T>
    template <typename S>
    requires(
        mstd::is_quantity_like_v<T> && !std::same_as<S, T> &&
        requires(T &elem, const S &scalar) { elem *= scalar; }
    )
    Vector3D<T> &Vector3D<T>::operator*=(const S &scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    /**
     * @brief component-wise operator *= for quantity elements and a vector of
     * dimensionless values
     *
     * @tparam T
     * @tparam S
     * @param vec
     * @return `Vector3D<T>&` (inplace multiplication)
     */
    template <typename T>
    template <typename S>
    requires(
        mstd::is_quantity_like_v<T> && !std::same_as<S, T> &&
        requires(T &elem, const S &scalar) { elem *= scalar; }
    )
    Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<S> &vec)
    {
        x *= vec[0];
        y *= vec[1];
        z *= vec[2];
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
     * @param index
     * @return T&
     */
    template <typename T>
    T &Vector3D<T>::operator[](size_t index)
    {
        return xyz[index];
    }

    /**
     * @brief const index operator
     *
     * @param index
     * @return const T&
     */
    template <typename T>
    const T &Vector3D<T>::operator[](size_t index) const
    {
        return xyz[index];
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
    template <typename T>
    Vector3D<T> Vector3D<T>::operator-() const
    requires pq::ArithmeticVector3D<T> || pq::Arithmetic<T> ||
             mstd::is_quantity_like_v<T>
    {
        return Vector3D<T>(-x, -y, -z);
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
    template <typename T>
    constexpr std::array<T, 3>::const_iterator Vector3D<T>::begin(
    ) const noexcept
    {
        return xyz.begin();
    }

    /**
     * @brief end iterator
     *
     * @return constexpr std::array<T, 3>::iterator
     */
    template <typename T>
    constexpr std::array<T, 3>::const_iterator Vector3D<T>::end() const noexcept
    {
        return xyz.end();
    }

    /*******************
     * casting methods *
     *******************/

    /**
     * @brief static cast of all vector members
     *
     * @tparam U
     * @return `Vector3D<U>`
     */
    template <typename T>
    template <typename U>
    Vector3D<T>::operator Vector3D<U>() const
    {
        return Vector3D<U>(
            static_cast<U>(x),
            static_cast<U>(y),
            static_cast<U>(z)
        );
    }

    /**
     * @brief cast to std::vector
     *
     * @return std::vector<T>
     */
    template <typename T>
    std::vector<T> Vector3D<T>::toStdVector() const
    {
        return std::vector<T>(xyz.begin(), xyz.end());
    }

    /**
     * @brief convert vector to specified unit
     *
     * @tparam U
     * @param unit
     * @return Vector3D<decltype(x * unit)>
     */
    template <typename T>
    template <mstd::UnitLike U>
    auto Vector3D<T>::in(const U &unit) const
    {
        return Vector3D<decltype(x.in(unit))>(
            x.in(unit),
            y.in(unit),
            z.in(unit)
        );
    }

    /**
     * @brief convert vector to numerical value in specified unit
     *
     * @tparam U
     * @param unit
     * @return Vector3D<decltype(x.numerical_value_in(unit))>
     */
    template <typename T>
    template <mstd::UnitLike U>
    auto Vector3D<T>::numerical_value_in(const U &unit) const
    {
        return Vector3D<decltype(x.numerical_value_in(unit))>(
            x.numerical_value_in(unit),
            y.numerical_value_in(unit),
            z.numerical_value_in(unit)
        );
    }

}   // namespace linalg

#endif   // LINEAR_ALGEBRA__VECTOR3D__VECTOR3D_CLASS_TPP
