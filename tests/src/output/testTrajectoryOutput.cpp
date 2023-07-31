#include "testTrajectoryOutput.hpp"

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
    EXPECT_EQ(line, "H         1.00000000     1.00000000     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O         1.00000000     2.00000000     3.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar        1.00000000     1.00000000     1.00000000");
}

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
    EXPECT_EQ(line, "H          1.00000000e+00      1.00000000e+00      1.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "O          3.00000000e+00      4.00000000e+00      5.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "Ar         1.00000000e+00      1.00000000e+00      1.00000000e+00");
}

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
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H         1.00000000     1.00000000     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O         2.00000000     3.00000000     4.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar        1.00000000     1.00000000     1.00000000");
}

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
    EXPECT_EQ(line, "H         1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O        -1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar        0.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}