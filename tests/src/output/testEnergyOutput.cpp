#include "testEnergyOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing energy output file
 *
 */
TEST_F(TestEnergyOutput, writeEnergyFile)
{
    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t      1.000000000000\t      2.000000000000\t      0.000000000000\t      3.000000000000\t      "
              "0.000000000000\t      "
              "4.000000000000\t      5.000000000000\t      0.000000000000\t      0.000000000000\t  6.000000000000e+00\t      "
              "0.000000000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}