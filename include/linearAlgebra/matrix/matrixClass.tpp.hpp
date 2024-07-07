#ifndef _MATRIX_CLASS_TPP_

#define _MATRIX_CLASS_TPP_

#include <cstddef>

#include "Eigen/Dense"
#include "exceptions.hpp"
#include "matrixClass.hpp"

namespace linearAlgebra
{

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
     * @brief index operator
     *
     * @param const size_t index
     * @return std::vector<T> &
     */
    template <typename T>
    T &Matrix<T>::operator()(const size_t index_i, const size_t index_j)
    {
        return _data(index_i, index_j);
    }

    /**
     * @brief index operator
     *
     * @param const size_t index
     * @return std::vector<T> &
     */
    template <typename T>
    std::vector<T> Matrix<T>::operator()(const size_t index)
    {
        std::vector<T> row;

        for (size_t i = 0; i < _cols; ++i) row.push_back(_data(index, i));

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
     * @tparam T
     * @return Matrix<T> &
     */
    template <typename T>
    Matrix<T> Matrix<T>::inverse()
    {
        return Matrix<T>(_data.inverse());
    }

    /**
     * @brief solve the linear system of equations
     *
     * @tparam T
     * @return Vector3D<T>
     *
     */
    template <typename T>
    std::vector<T> Matrix<T>::solve(const std::vector<T> &b)
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> bEigen(b.size());
        std::ranges::copy(b, bEigen.data());

        Eigen::MatrixXd              matrix = _data.transpose() * _data;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(matrix);

        if (ldlt.info() != Eigen::Success)
        {
            // matrix is not positive-definite
            throw customException::LinearAlgebraException(
                "Matrix is not positive-definite."
            );
        }
        auto solutionEigen = ldlt.solve(_data.transpose() * bEigen);

        std::vector<T> solution;
        for (int i = 0; i < solutionEigen.size(); ++i)
            solution.push_back(solutionEigen(i));

        return solution;
    }
}   // namespace linearAlgebra

#endif   // _MATRIX_CLASS_TPP_