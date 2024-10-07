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

#ifndef _MATRIX_CLASS_HPP_

#define _MATRIX_CLASS_HPP_

#include <cstddef>

#include "Eigen/Dense"

namespace linearAlgebra
{
    /**
     * @class Matrix
     *
     */
    template <typename T>
    class Matrix
    {
       protected:
        size_t                                           _rows;
        size_t                                           _cols;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _data;

       public:
        Matrix() = default;
        explicit Matrix(const size_t rows, const size_t cols);
        explicit Matrix(const size_t rowsAndCols);
        explicit Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data);

        [[nodiscard]] T &operator()(const size_t index_i, const size_t index_j);
        [[nodiscard]] std::vector<T> operator()(const size_t index);

        [[nodiscard]] std::pair<size_t, size_t> shape() const;

        [[nodiscard]] size_t rows() const { return _rows; }
        [[nodiscard]] size_t cols() const { return _cols; }
        [[nodiscard]] size_t size() const { return _rows * _cols; }

        Matrix<T>      inverse();
        std::vector<T> solve(const std::vector<T> &b);
    };

}   // namespace linearAlgebra

#include "matrixClass.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _MATRIX_CLASS_HPP_