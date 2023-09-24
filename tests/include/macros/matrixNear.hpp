#ifndef _EXPECT_MATRIX_NEAR_HPP_

#define _EXPECT_MATRIX_NEAR_HPP_

#define EXPECT_MATRIX_NEAR(matrix1, matrix2, value)                                                                              \
    EXPECT_NEAR(matrix1[0][0], matrix2[0][0], value);                                                                            \
    EXPECT_NEAR(matrix1[0][1], matrix2[0][1], value);                                                                            \
    EXPECT_NEAR(matrix1[0][2], matrix2[0][2], value);                                                                            \
    EXPECT_NEAR(matrix1[1][0], matrix2[1][0], value);                                                                            \
    EXPECT_NEAR(matrix1[1][1], matrix2[1][1], value);                                                                            \
    EXPECT_NEAR(matrix1[1][2], matrix2[1][2], value);                                                                            \
    EXPECT_NEAR(matrix1[2][0], matrix2[2][0], value);                                                                            \
    EXPECT_NEAR(matrix1[2][1], matrix2[2][1], value);                                                                            \
    EXPECT_NEAR(matrix1[2][2], matrix2[2][2], value);

#define ASSERT_MATRIX_NEAR(matrix1, matrix2) EXPECT_MATRIX_NEAR(matrix1, matrix2)

#endif   // _EXPECT_MATRIX_DOUBLE_EQ_HPP_