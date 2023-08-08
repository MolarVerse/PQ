#ifndef _STATIC_MATRIX_HPP_

#define _STATIC_MATRIX_HPP_

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iostream>
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
    size_t                      _rows;
    size_t                      _cols;
    std::vector<std::vector<T>> _data;

  public:
    Matrix() = default;
    explicit Matrix(const size_t rows, const size_t cols) : _rows(rows), _cols(cols)
    {
        _data.resize(rows);
        std::ranges::for_each(_data, [cols](auto &row) { row.resize(cols); });
    }
    explicit Matrix(const size_t rowsAndCols) : _rows(rowsAndCols), _cols(rowsAndCols)
    {
        _data.resize(rowsAndCols);
        std::ranges::for_each(_data, [rowsAndCols](auto &row) { row.resize(rowsAndCols); });
    }

    /**
     * @brief index operator
     *
     * @param const size_t index
     * @return std::vector<T> &
     */
    std::vector<T> &operator[](const size_t index) { return _data[index]; }

    [[nodiscard]] size_t                    rows() const { return _rows; }
    [[nodiscard]] size_t                    cols() const { return _cols; }
    [[nodiscard]] size_t                    size() const { return _rows * _cols; }
    [[nodiscard]] std::pair<size_t, size_t> shape() const { return std::make_pair(_rows, _cols); }
};

#endif   // _STATIC_MATRIX_HPP_