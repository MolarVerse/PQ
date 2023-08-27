#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for RstFileException, customException
#include "restartFileSection.hpp"       // for RstFileSection, readInput
#include "simulationBox.hpp"            // for SimulationBox
#include "simulationBoxSettings.hpp"    // for SimulationBoxSettings
#include "testRestartFileSection.hpp"   // for TestBoxSection
#include "vector3d.hpp"                 // for linearAlgebra

#include "gmock/gmock.h"   // for ElementsAre, MakePredicateForma...
#include "gtest/gtest.h"   // for Message, TestPartResult, Assert...
#include <algorithm>       // for copy
#include <cstddef>         // for size_t, std
#include <gtest/gtest.h>   // for TestInfo (ptr only), ASSERT_THROW
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace customException;
using namespace std;
using namespace testing;
using namespace readInput;
using namespace linearAlgebra;

TEST_F(TestBoxSection, testKeyword) { EXPECT_EQ(_section->keyword(), "box"); }

TEST_F(TestBoxSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestBoxSection, testNumberOfArguments)
{
    for (size_t i = 0; i < 10; ++i)
        if (i != 4 && i != 7)
        {
            auto line = vector<string>(i);
            ASSERT_THROW(_section->process(line, _engine), RstFileException);
        }
}

TEST_F(TestBoxSection, testProcess)
{
    EXPECT_EQ(settings::SimulationBoxSettings::getBoxSet(), false);

    vector<string> line = {"box", "1.0", "2.0", "3.0"};
    _section->process(line, _engine);
    ASSERT_THAT(_engine.getSimulationBox().getBoxDimensions(), ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine.getSimulationBox().getBoxAngles(), ElementsAre(90.0, 90.0, 90.0));

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "70.0"};
    _section->process(line, _engine);
    ASSERT_THAT(_engine.getSimulationBox().getBoxDimensions(), ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine.getSimulationBox().getBoxAngles(), ElementsAre(90.0, 90.0, 70.0));

    line = {"box", "1.0", "2.0", "-3.0", "90.0", "90.0", "90.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "190.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "-90.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    EXPECT_EQ(settings::SimulationBoxSettings::getBoxSet(), true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}