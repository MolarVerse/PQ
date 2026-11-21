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

#ifndef LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_HPP
#define LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <mstd/type_traits/quantity_traits.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace linalg
{
    // element type of the inverse, e.g. Pos^2/Mass -> Mass/Pos^2
    // (falls back to T for element types without a division, e.g. pointers,
    // so that Matrix<T> can still be instantiated for them)
    template <typename T>
    struct InverseOfImpl
    {
        using type = T;
    };

    template <typename T>
    requires requires(const T &elem) { 1.0 / elem; }
    struct InverseOfImpl<T>
    {
        using type = std::remove_cvref_t<decltype(1.0 / std::declval<T>())>;
    };

    template <typename T>
    using InverseOf = typename InverseOfImpl<T>::type;

    /**
     * @class Matrix
     *
     */
    template <typename T>
    class Matrix
    {
       private:
        size_t _rows = 0;
        size_t _cols = 0;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _data;

       public:
        Matrix() = default;
        explicit Matrix(size_t rows, size_t cols);
        explicit Matrix(size_t rowsAndCols);
        explicit Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data);

        [[nodiscard]] T             &operator()(size_t index_i, size_t index_j);
        [[nodiscard]] std::vector<T> operator()(size_t index);

        [[nodiscard]] std::pair<size_t, size_t> shape() const;

        [[nodiscard]] size_t rows() const { return _rows; }
        [[nodiscard]] size_t cols() const { return _cols; }
        [[nodiscard]] size_t size() const { return _rows * _cols; }

        Matrix<InverseOf<T>> inverse();

        // solves A * x = rhs; the element type of x is the quotient type
        // rhs[0] / A(0, 0), e.g. Pos^2/Time^2 / (Pos^2/Mass) -> Mass/Time^2
        template <typename U>
        auto solve(const std::vector<U> &rhs)
            -> std::vector<std::remove_cvref_t<
                decltype(std::declval<U>() / std::declval<T>())>>;
    };

}   // namespace linalg

#ifndef LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_TPP
#include "matrixClass.tpp"   // IWYU pragma: export
#endif

#endif   // LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_HPP
