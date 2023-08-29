#include "testParameterFileSection.hpp"

#include "bondSection.hpp"   // for BondSection

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <ostream>         // for operator<<, ofstream, basic_ostream, endl
#include <vector>          // for vector

/**
 * @brief tests full process function TODO: think of a clever way to test this
 *
 */
TEST_F(TestParameterFileSection, processParameterSection)
{
    readInput::parameterFile::BondSection section;

    std::ofstream outputStream(_parameterFileName.c_str());

    outputStream << "bonds\n";
    outputStream << "1 2 1.0\n";
    outputStream << "         \n";
    outputStream << "2 3 1.2\n";
    outputStream << "end" << '\n' << std::flush;

    outputStream.close();

    auto          lineElements = std::vector{std::string("")};
    std::ifstream fp(_parameterFileName.c_str());
    getline(fp, lineElements[0]);

    section.setFp(&fp);
    section.setLineNumber(1);

    EXPECT_NO_THROW(section.process(lineElements, *_engine));

    EXPECT_EQ(section.getLineNumber(), 5);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}