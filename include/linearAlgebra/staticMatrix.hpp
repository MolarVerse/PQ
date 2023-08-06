#ifndef _STATIC_MATRIX_HPP_

#define _STATIC_MATRIX_HPP_

#include <concepts>
#include <cstddef>
#include <vector>

namespace linearAlgebra
{
    template <typename T>
    class Matrix;
}   // namespace linearAlgebra

/**
 * @class Matrix
 *
 * @brief Matrix is a class for static matrix
 *
 */
template <typename T>
class linearAlgebra::Matrix
{
  protected:
    size_t         _rows;
    size_t         _cols;
    std::vector<T> _data;

  public:
    Matrix() = default;
    explicit Matrix(const std::vector<T> &vec) : _data(vec){};
    explicit Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), _data(rows * cols){};
    explicit Matrix(size_t rowsAndCols) : _rows(rowsAndCols), _cols(rowsAndCols), _data(rowsAndCols * rowsAndCols){};

    /**
     * @brief index operator
     *
     * @param const size_t index
     * @return T&
     */
    T &operator[](const size_t index)
    {
        const auto start = _data.begin() + _cols * index;
        const auto end   = _data.begin() + _cols * 2 * index;
        return std::vector<T>(start, end);
    }

    Matrix<T> transpose() const
    {
        Matrix<T> mat(_cols, _rows);

        for (size_t i = 0; i < _rows; ++i)
            for (size_t j = 0; j < _cols; ++j)
                mat[j][i] = _data[i][j];

        return mat;
    }

    bool isSymmetric(const linearAlgebra::Matrix<T> &mat) const
    requires std::equality_comparable<T>
    {
        if (_rows != _cols) return false;

        for (size_t i = 0; i < _rows; ++i)
            if (mat[i] != mat.transpose()[i]) return false;

        return true;
    }
};

#endif   // _STATIC_MATRIX_HPP_