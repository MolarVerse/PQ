#ifndef _STATIC_MATRIX_CLASS_3X3_HPP_

#define _STATIC_MATRIX_CLASS_3X3_HPP_

#include "vector3d.hpp"

namespace linearAlgebra
{

    /**
     * @class StaticMatrix3x3
     *
     * @brief template Matrix class with 3 rows and 3 columns
     *
     * @tparam T
     */
    template <typename T>
    class StaticMatrix3x3
    {
      private:
        Vector3D<Vector3D<T>> _data;

      public:
        StaticMatrix3x3() = default;

        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &data) : _data(data) {}
        explicit StaticMatrix3x3(const Vector3D<Vector3D<T>> &&data) : _data(std::move(data)) {}

        explicit StaticMatrix3x3(const Vector3D<T> &row1, const Vector3D<T> &row2, const Vector3D<T> &row3)
            : _data(row1, row2, row3){};

        explicit StaticMatrix3x3(const T t)
        {
            _data[0] = Vector3D<T>(t);
            _data[1] = Vector3D<T>(t);
            _data[2] = Vector3D<T>(t);
        }

        /**
         * @brief index operator
         *
         * @param const size_t index
         * @return std::vector<T> &
         */
        Vector3D<T> &operator[](const size_t index) { return _data[index]; }

        /**
         * @brief index operator
         *
         * @param const size_t index
         * @return const std::vector<T> &
         */
        const Vector3D<T> &operator[](const size_t index) const { return _data[index]; }

        /**
         * @brief operator== for two StaticMatrix3x3's
         *
         * @param rhs
         * @return bool
         */
        friend bool operator==(const StaticMatrix3x3 &lhs, const StaticMatrix3x3 &rhs) { return lhs._data == rhs._data; }
    };
}   // namespace linearAlgebra

#endif   // _STATIC_MATRIX_CLASS_3X3_HPP_