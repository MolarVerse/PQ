#include "testLogOutput.hpp"

TEST_F(TestLogOutput, writeDensityWarning)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeDensityWarning();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "WARNING: Density and box dimensions set. Density will be ignored.");
}

TEST_F(TestLogOutput, writeInitialMomentum)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeInitialMomentum(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    getline(file, line);
    EXPECT_EQ(line, "Initial momentum = 0.100000 Angstrom * amu / fs");
}

TEST_F(TestLogOutput, writeRelaxationTimeThermostatWarning)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeRelaxationTimeThermostatWarning();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "WARNING: Berendsen thermostat set but no relaxation time given. Using default value of 0.1ps.");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}