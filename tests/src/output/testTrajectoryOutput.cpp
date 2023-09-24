#include "testTrajectoryOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestTrajectoryOutput, writeXyz)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeXyz(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t     1.00000000\t     2.00000000\t     3.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     1.00000000\t     1.00000000\t     1.00000000");
}

/**
 * @brief Test the writeVelocities method
 *
 */
TEST_F(TestTrajectoryOutput, writeVelocities)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeVelocities(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "O    \t      3.00000000e+00\t      4.00000000e+00\t      5.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
}

/**
 * @brief Test the writeForces method
 *
 */
TEST_F(TestTrajectoryOutput, writeForces)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeForces(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "# Total force = 8.77496e+00 kcal/mol/Angstrom");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t     2.00000000\t     3.00000000\t     4.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     1.00000000\t     1.00000000\t     1.00000000");
}

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestTrajectoryOutput, writeCharges)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeCharges(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t    -1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     0.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}