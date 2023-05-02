#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <cassert>

#include "testRstFileSection.hpp"

using namespace std;
using namespace testing;

TEST_F(TestRstFileSection, testBoxSection)
{
    vector<string> line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "90.0"};
    _section->process(line, _settings, _simulationBox);
    EXPECT_EQ(_section->keyword(), "box");
    ASSERT_THAT(_simulationBox._box.getBoxDimensions(), ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_simulationBox._box.getBoxAngles(), ElementsAre(90.0, 90.0, 90.0));

    for (int i = 0; i < 10; ++i)
    {
        if (i != 4 || i != 7)
        {
            line = vector<string>(i);
            ASSERT_THROW(_section->process(line, _settings, _simulationBox), invalid_argument);
        }
    }

    line = {"box", "1.0", "2.0", "-3.0", "90.0", "90.0", "90.0"};
    ASSERT_THROW(_section->process(line, _settings, _simulationBox), range_error);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "190.0"};
    ASSERT_THROW(_section->process(line, _settings, _simulationBox), range_error);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "-90.0"};
    ASSERT_THROW(_section->process(line, _settings, _simulationBox), range_error);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
