#ifndef _STATIC_MATRIX_CLASS_3x3_TPP_

#define _STATIC_MATRIX_CLASS_3x3_TPP_

#include "exceptions.hpp"
#include "staticMatrix3x3Class.hpp"

namespace linearAlgebra
{
    /**
     * @brief Construct a new Static Matrix 3x 3< T>:: Static Matrix 3x 3 object
     *
     * @tparam T
     * @param data
     */
    template <typename T>
    StaticMatrix3x3<T>::StaticMatrix3x3(const Vector3D<Vector3D<T>> &data)
        : _data(data)
    {
    }

    /**
     * @brief Construct a new Static Matrix 3x 3< T>:: Static Matrix 3x 3 object
     *
     * @tparam T
     * @param data
     */
    template <typename T>
    StaticMatrix3x3<T>::StaticMatrix3x3(const Vector3D<Vector3D<T>> &&data)
        : _data(std::move(data))
    {
    }

    /**
     * @brief Construct a new Static Matrix 3x 3< T>:: Static Matrix 3x 3 object
     *
     * @tparam T
     * @param row1
     * @param row2
     * @param row3
     */
    template <typename T>
    StaticMatrix3x3<T>::StaticMatrix3x3(
        const Vector3D<T> &row1,
        const Vector3D<T> &row2,
        const Vector3D<T> &row3
    )
        : _data(row1, row2, row3){};

    /**
     * @brief Construct a new Static Matrix 3x 3< T>:: Static Matrix 3x 3 object
     *
     * @tparam T
     * @param t
     */
    template <typename T>
    StaticMatrix3x3<T>::StaticMatrix3x3(const T t)
    {
        _data[0] = Vector3D<T>(t);
        _data[1] = Vector3D<T>(t);
        _data[2] = Vector3D<T>(t);
    }

    /**
     * @brief Construct a new Static Matrix 3x 3< T>:: Static Matrix 3x 3 object
     *
     * @tparam T
     * @param vector
     */
    template <typename T>
    StaticMatrix3x3<T>::StaticMatrix3x3(const std::vector<T> &vector)
    {
        if (vector.size() != 9)
            throw customException::LinearAlgebraException(
                "vector size must be 9"
            );

        _data[0] = Vector3D<T>(vector[0], vector[1], vector[2]);
        _data[1] = Vector3D<T>(vector[3], vector[4], vector[5]);
        _data[2] = Vector3D<T>(vector[6], vector[7], vector[8]);
    }

    /**
     * @brief index operator
     *
     * @tparam T
     * @param const size_t index
     * @return std::vector<T> &
     */
    template <typename T>
    Vector3D<T> &StaticMatrix3x3<T>::operator[](const size_t index)
    {
        return _data[index];
    }

    /**
     * @brief index operator
     *
     * @tparam T
     * @param const size_t index
     * @return const std::vector<T> &
     */
    template <typename T>
    const Vector3D<T> &StaticMatrix3x3<T>::operator[](const size_t index) const
    {
        return _data[index];
    }

    /**
     * @brief unary operator- for StaticMatrix3x3
     *
     * @tparam T
     * @return StaticMatrix3x3
     */
    template <typename T>
    StaticMatrix3x3<T> StaticMatrix3x3<T>::operator-()
    {
        return StaticMatrix3x3(-_data[0], -_data[1], -_data[2]);
    }

    /**
     * @brief operator+ for two StaticMatrix3x3's
     *
     * @tparam T
     * @param rhs
     * @return StaticMatrix3x3
     */
    template <typename T>
    std::vector<T> StaticMatrix3x3<T>::toStdVector() const
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
}   // namespace linearAlgebra

#endif   // _STATIC_MATRIX_CLASS_3x3_TPP_