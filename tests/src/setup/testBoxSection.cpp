#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <cassert>

#include "testRstFileSection.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace testing;

TEST_F(TestBoxSection, testKeyword)
{
    EXPECT_EQ(_section->keyword(), "box");
}

TEST_F(TestBoxSection, testIsHeader)
{
    EXPECT_TRUE(_section->isHeader());
}

TEST_F(TestBoxSection, testNumberOfArguments)
{
    for (int i = 0; i < 10; ++i)
        if (i != 4 && i != 7)
        {
            auto line = vector<string>(i);
            ASSERT_THROW(_section->process(line, _engine), RstFileException);
        }
}

TEST_F(TestBoxSection, testProcess)
{
    vector<string> line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "70.0"};
    _section->process(line, _engine);
    ASSERT_THAT(_engine._simulationBox._box.getBoxDimensions(), ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine._simulationBox._box.getBoxAngles(), ElementsAre(90.0, 90.0, 70.0));

    line = {"box", "1.0", "2.0", "3.0"};
    _section->process(line, _engine);
    ASSERT_THAT(_engine._simulationBox._box.getBoxDimensions(), ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine._simulationBox._box.getBoxAngles(), ElementsAre(90.0, 90.0, 90.0));

    line = {"box", "1.0", "2.0", "-3.0", "90.0", "90.0", "90.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "190.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "-90.0"};
    ASSERT_THROW(_section->process(line, _engine), RstFileException);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
