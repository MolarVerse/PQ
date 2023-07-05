#include "testEnergyOutput.hpp"

TEST_F(TestEnergyOutput, writeEnergyFile)
{
    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100      1.000000000000      2.000000000000      0.000000000000      3.000000000000      0.000000000000      "
        "4.000000000000      5.000000000000      0.000000000000      0.000000000000  6.000000000000e+00      0.000000000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}