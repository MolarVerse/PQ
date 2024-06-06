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

#ifndef _MATRIX_HPP_

#define _MATRIX_HPP_

#include <algorithm>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <utility>   // for make_pair, pair
#include <vector>

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

        /**
         * @brief index operator
         *
         * @param const size_t index
         * @return std::vector<T> &
         */
        T &operator()(const size_t index_i, const size_t index_j)
        {
            return _data(index_i, index_j);
        }

        /**
         * @brief const index operator
         *
         * @param const size_t index
         * @return const std::vector<T> &
         */
        Matrix<T> inverse();

        [[nodiscard]] size_t rows() const { return _rows; }
        [[nodiscard]] size_t cols() const { return _cols; }
        [[nodiscard]] size_t size() const { return _rows * _cols; }
        [[nodiscard]] std::pair<size_t, size_t> shape() const
        {
            return std::make_pair(_rows, _cols);
        }
    };

    /**
     * @brief Construct a new Matrix< T>:: Matrix object
     *
     * @tparam T
     * @param rows
     * @param cols
     */
    template <typename T>
    Matrix<T>::Matrix(const size_t rows, const size_t cols)
        : _rows(rows), _cols(cols)
    {
        _data.resize(rows, cols);
    }

    /**
     * @brief Construct a new Matrix< T>:: Matrix object
     *
     * @tparam T
     * @param rowsAndCols
     */
    template <typename T>
    Matrix<T>::Matrix(const size_t rowsAndCols)
        : _rows(rowsAndCols), _cols(rowsAndCols)
    {
        _data.resize(rowsAndCols, rowsAndCols);
    }

    /**
     * @brief Construct a new Matrix< T>:: Matrix object
     *
     * @tparam T
     * @param data
     */
    template <typename T>
    Matrix<T>::Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data)
        : _data(data)
    {
        _rows = _data.rows();
        _cols = _data.cols();
    }

    /**
     * @brief Inverts the matrix using Eigen library
     *
     * @tparam T
     * @return Matrix<T> &
     */
    template <typename T>
    Matrix<T> Matrix<T>::inverse()
    {
        return Matrix<T>(_data.inverse());
    }

}   // namespace linearAlgebra

#endif   // _MATRIX_HPP_