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