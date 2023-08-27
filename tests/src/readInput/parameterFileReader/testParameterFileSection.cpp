#include "testParameterFileSection.hpp"

#include "angleSection.hpp"
#include "bondSection.hpp"
#include "dihedralSection.hpp"
#include "exceptions.hpp"
#include "improperDihedralSection.hpp"
#include "nonCoulombicsSection.hpp"
#include "parameterFileSection.hpp"

/**
 * @brief tests full process function TODO: think of a clever way to test this
 *
 */
TEST_F(TestParameterFileSection, processParameterSection)
{
    readInput::parameterFile::BondSection section;

    std::ofstream _outputStream(_parameterFilename.c_str());

    _outputStream << "bonds\n";
    _outputStream << "1 2 1.0\n";
    _outputStream << "         \n";
    _outputStream << "2 3 1.2\n";
    _outputStream << "end" << std::endl;

    _outputStream.close();

    auto          lineElements = std::vector{std::string("")};
    std::ifstream fp(_parameterFilename.c_str());
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