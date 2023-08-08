#include "testStdoutOutput.hpp"

/**
 * @brief Test writeDensityWarning
 *
 */
TEST_F(TestStdoutOutput, writeDensityWarning)
{
    testing::internal::CaptureStdout();
    _stdoutOutput->writeDensityWarning();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "\x1B[33mUserInputWarning\x1B[39m\nDensity and box dimensions set. Density will be ignored.\n\n");
}

/**
 * @brief Test writeInitialMomentum
 *
 */
TEST_F(TestStdoutOutput, writeInitialMomentum)
{
    testing::internal::CaptureStdout();
    _stdoutOutput->writeInitialMomentum(0.1);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "\nInitial momentum = 0.100000 Angstrom * amu / fs\n");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}