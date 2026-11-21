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

#ifndef LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_TPP
#define LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_TPP

#include <cstddef>
#include <cstdint>

#include "Eigen/Dense"
#include "exceptions.hpp"
#include "matrixClass.hpp"

namespace linalg
{

    /**
     * @brief Construct a new Matrix< T>:: Matrix object
     *
     * @tparam T
     * @param rows
     * @param cols
     */
    template <typename T>
    Matrix<T>::Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols)
    {
        _data.resize(
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols)
        );
    }

    /**
     * @brief Construct a new Matrix< T>:: Matrix object
     *
     * @tparam T
     * @param rowsAndCols
     */
    template <typename T>
    Matrix<T>::Matrix(size_t rowsAndCols)
        : _rows(rowsAndCols), _cols(rowsAndCols)
    {
        _data.resize(
            static_cast<std::int64_t>(rowsAndCols),
            static_cast<std::int64_t>(rowsAndCols)
        );
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
        _rows = static_cast<size_t>(_data.rows());
        _cols = static_cast<size_t>(_data.cols());
    }

    /**
     * @brief index operator
     *
     * @param index_i
     * @param index_j
     * @return `T &`
     */
    template <typename T>
    T &Matrix<T>::operator()(size_t index_i, size_t index_j)
    {
        return _data(
            static_cast<std::int64_t>(index_i),
            static_cast<std::int64_t>(index_j)
        );
    }

    /**
     * @brief index operator
     *
     * @param index
     * @return `std::vector<T> &`
     */
    template <typename T>
    std::vector<T> Matrix<T>::operator()(size_t index)
    {
        std::vector<T> row;

        for (size_t i = 0; i < _cols; ++i)
        {
            row.push_back(_data(
                static_cast<std::int64_t>(index),
                static_cast<std::int64_t>(i)
            ));
        }

        return row;
    }

    /**
     * @brief returns the shape of the matrix
     *
     * @return std::pair<size_t, size_t>
     */
    template <typename T>
    std::pair<size_t, size_t> Matrix<T>::shape() const
    {
        return std::make_pair(_rows, _cols);
    }

    /**
     * @brief Inverts the matrix using Eigen library
     *
     * @details The units are handled at the Eigen boundary: the numerical
     * values are inverted and the result carries the inverse unit of the
     * matrix elements (1 / T).
     *
     * @tparam T
     * @return Matrix<decltype(1.0 / T)>
     */
    template <typename T>
    Matrix<InverseOf<T>> Matrix<T>::inverse()
    {
        using Result = InverseOf<T>;

        auto numeric = []<typename V>(const V &value) -> double
        {
            if constexpr (mstd::is_quantity_like_v<V>)
                return value.numerical_value_in(V::unit);
            else
                return static_cast<double>(value);
        };

        Eigen::MatrixXd values(_data.rows(), _data.cols());
        for (Eigen::Index i = 0; i < _data.rows(); ++i)
            for (Eigen::Index j = 0; j < _data.cols(); ++j)
                values(i, j) = numeric(_data(i, j));

        const Eigen::MatrixXd inverseValues = values.inverse();

        const auto resultUnit = [&]
        {
            if constexpr (mstd::is_quantity_like_v<Result>)
                return Result::unit;
            else
                return 1.0;
        }();

        Matrix<Result> result(_rows, _cols);
        for (Eigen::Index i = 0; i < inverseValues.rows(); ++i)
            for (Eigen::Index j = 0; j < inverseValues.cols(); ++j)
                result(static_cast<size_t>(i), static_cast<size_t>(j)) =
                    inverseValues(i, j) * resultUnit;

        return result;
    }

    /**
     * @brief solve the linear system of equations A * x = rhs in the least
     * squares sense (normal equations), where the elements of A and rhs may
     * be quantities
     *
     * @details The units are handled at the Eigen boundary: the numerical
     * values (in the units of A and rhs) are solved for and the result is
     * returned with the unit of the quotient rhs / A.
     *
     * @tparam T element type of the matrix
     * @tparam U element type of the right hand side
     * @param rhs
     * @return `std::vector<decltype(U / T)>`
     */
    template <typename T>
    template <typename U>
    auto Matrix<T>::solve(const std::vector<U> &rhs) -> std::vector<
        std::remove_cvref_t<decltype(std::declval<U>() / std::declval<T>())>>
    {
        using Result = std::remove_cvref_t<
            decltype(std::declval<U>() / std::declval<T>())>;

        // numerical value of a (possibly unit carrying) element
        auto numeric = []<typename V>(const V &value) -> double
        {
            if constexpr (mstd::is_quantity_like_v<V>)
                return value.numerical_value_in(V::unit);
            else
                return static_cast<double>(value);
        };

        Eigen::MatrixXd matrixValues(_data.rows(), _data.cols());
        for (Eigen::Index i = 0; i < _data.rows(); ++i)
            for (Eigen::Index j = 0; j < _data.cols(); ++j)
                matrixValues(i, j) = numeric(_data(i, j));

        Eigen::VectorXd bEigen(static_cast<Eigen::Index>(rhs.size()));
        for (size_t i = 0; i < rhs.size(); ++i)
            bEigen(static_cast<Eigen::Index>(i)) = numeric(rhs[i]);

        Eigen::MatrixXd matrix = matrixValues.transpose() * matrixValues;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(matrix);

        if (ldlt.info() != Eigen::Success)
        {
            // matrix is not positive-definite
            throw exc::LinearAlgebraException(
                "Matrix is not positive-definite."
            );
        }

        const Eigen::VectorXd solutionEigen =
            ldlt.solve(matrixValues.transpose() * bEigen);

        // the unit of x is the unit of rhs / A
        const auto resultUnit = [&]
        {
            if constexpr (mstd::is_quantity_like_v<Result>)
                return Result::unit;
            else
                return 1.0;
        }();

        std::vector<Result> solution;
        solution.reserve(static_cast<size_t>(solutionEigen.size()));
        for (Eigen::Index i = 0; i < solutionEigen.size(); ++i)
            solution.push_back(solutionEigen(i) * resultUnit);

        return solution;
    }
}   // namespace linalg

#endif   // LINEAR_ALGEBRA__MATRIX__MATRIX_CLASS_TPP
