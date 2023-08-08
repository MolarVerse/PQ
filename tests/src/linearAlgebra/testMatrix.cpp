#include "matrix.hpp"

#include <gtest/gtest.h>

using namespace linearAlgebra;

/**
 * @brief tests constructors for Matrix
 *
 */
TEST(TestMatrix, constructors)
{
    Matrix<int> mat(2);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);

    Matrix<int> mat2(2, 3);
    EXPECT_EQ(mat2.rows(), 2);
    EXPECT_EQ(mat2.cols(), 3);
}

/**
 * @brief tests shape function for Matrix
 *
 */
TEST(TestMatrix, shape)
{
    auto mat = Matrix<int>(2, 3);

    const auto [rows, cols] = mat.shape();
    EXPECT_EQ(rows, 2);
    EXPECT_EQ(cols, 3);
}

/**
 * @brief tests size function for Matrix
 *
 */
TEST(TestMatrix, size)
{
    auto mat = Matrix<int>(2, 3);

    EXPECT_EQ(mat.size(), 6);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}