#include "testLogOutput.hpp"

/**
 * @brief tests writing density warning to log file
 *
 */
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

/**
 * @brief tests writing initial momentum to log file
 *
 */
TEST_F(TestLogOutput, writeInitialMomentum)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeInitialMomentum(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    getline(file, line);
    EXPECT_EQ(line, "Initial momentum = 0.1 Angstrom * amu / fs");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}