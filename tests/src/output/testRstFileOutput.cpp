#include "testRstFileOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing the restart file
 *
 */
TEST_F(TestRstFileOutput, write)
{
    _rstFileOutput->setFilename("default.rst");
    _rstFileOutput->write(*_simulationBox, 10);
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line,
              "H    1    1         1.00000000     1.00000000     1.00000000      1.00000000e+00      1.00000000e+00      "
              "1.00000000e+00     1.00000000     1.00000000     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "O    2    1         1.00000000     2.00000000     3.00000000      3.00000000e+00      4.00000000e+00      "
              "5.00000000e+00     2.00000000     3.00000000     4.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "Ar   1    2         1.00000000     1.00000000     1.00000000      1.00000000e+00      1.00000000e+00      "
              "1.00000000e+00     1.00000000     1.00000000     1.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}