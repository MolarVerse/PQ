#ifndef _MATRIX_NX3_HPP_

#define _MATRIX_NX3_HPP_

#include "matrix.hpp"

#include <vector>

namespace linearAlgebra
{
    /**
     * @class MatrixNX3
     *
     * @brief template Matrix class with 3 columns
     *
     * @tparam T
     */
    template <typename T>
    class MatrixNX3 : public Matrix<T>
    {
      public:
        MatrixNX3() = default;
        explicit MatrixNX3(const size_t rows) : Matrix(rows, 3) {}

        [[nodiscard]] size_t                    cols() const { return 3; }
        [[nodiscard]] size_t                    size() const { return _rows * 3; }
        [[nodiscard]] std::pair<size_t, size_t> shape() const { return std::make_pair(_rows, 3); }
    };

}   // namespace linearAlgebra

#endif   // _MATRIX_NX3_HPP_