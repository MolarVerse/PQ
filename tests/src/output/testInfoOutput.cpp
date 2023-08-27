#include "infoOutput.hpp"         // for InfoOutput
#include "testEnergyOutput.hpp"   // for TestEnergyOutput

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, InitGoogleTest, RUN_ALL_T...
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing info file
 *
 */
TEST_F(TestEnergyOutput, writeInfo)
{
    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                PIMD-QMCF info file                                 |");
    getline(file, line);
    EXPECT_EQ(line, "-------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE         1.00000 K           |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)              0.00000 kcal/mol    |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)            0.00000 kcal/mol    |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)         5.00000 kcal/mol    |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                6.0e+00 amuA/fs  LOOPTIME            0.00000 s           |");
    getline(file, line);
    EXPECT_EQ(line, "-------------------------------------------------------------------------------------");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}