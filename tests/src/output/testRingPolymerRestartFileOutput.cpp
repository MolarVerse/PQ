#include "testRingPolymerRestartFileOutput.hpp"

#include "ringPolymerSettings.hpp"   // for RingPolymerSettings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, InitGoogleTest, RUN_ALL_TESTS
#include <iosfwd>          // for ifstream
#include <string>          // for getline, string

/**
 * @brief tests writing restart file for ring polymer
 *
 */
TEST_F(TestRingPolymerRestartFileOutput, write)
{
    settings::RingPolymerSettings::setNumberOfBeads(_beads.size());

    _rstFileOutput->setFilename("default.rpmd.rst");
    _rstFileOutput->write(_beads, 10);
    _rstFileOutput->close();
    std::ifstream file("default.rpmd.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line,
              "    H1\t    1\t    1\t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    O1\t    2\t    1\t     1.00000000\t     2.00000000\t     3.00000000\t     3.00000000e+00\t     "
              "4.00000000e+00\t     5.00000000e+00\t     2.00000000\t     3.00000000\t     4.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "   Ar1\t    1\t    2\t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    H2\t    1\t    1\t     2.00000000\t     2.00000000\t     2.00000000\t     2.00000000e+00\t     "
              "2.00000000e+00\t     2.00000000e+00\t     2.00000000\t     2.00000000\t     2.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    O2\t    2\t    1\t     2.00000000\t     3.00000000\t     4.00000000\t     4.00000000e+00\t     "
              "5.00000000e+00\t     6.00000000e+00\t     3.00000000\t     4.00000000\t     5.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "   Ar2\t    1\t    2\t     2.00000000\t     2.00000000\t     2.00000000\t     2.00000000e+00\t     "
              "2.00000000e+00\t     2.00000000e+00\t     2.00000000\t     2.00000000\t     2.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}