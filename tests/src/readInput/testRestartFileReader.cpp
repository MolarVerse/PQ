#include "testRestartFileReader.hpp"

#include "fileSettings.hpp"          // for FileSettings
#include "moldescriptorReader.hpp"   // for MoldescriptorReader
#include "restartFileReader.hpp"     // for RstFileReader, readRstFile
#include "restartFileSection.hpp"    // for RstFileSection, readInput

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests determineSection base on the first element of the line
 *
 */
TEST_F(TestRstFileReader, determineSection)
{
    std::string                    filename = "examples/setup/h2o_qmcfc.rst";
    restartFile::RestartFileReader rstFileReader(filename, _engine);

    auto  lineElements = std::vector<std::string>{"sTeP", "1"};
    auto *section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "step");

    lineElements = std::vector<std::string>{"Box"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "box");

    lineElements = std::vector<std::string>{"notAHeaderSection"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "");
}

/**
 * @brief test full read restart file function
 *
 */
TEST_F(TestRstFileReader, rstFileReading)
{
    settings::FileSettings::setMolDescriptorFileName("examples/setup/moldescriptor.dat");
    molDescriptor::MoldescriptorReader moldescriptor(_engine);

    std::string filename = "examples/setup/h2o-qmcf.rst";
    settings::FileSettings::setStartFileName(filename);

    moldescriptor.read();
    ASSERT_NO_THROW(restartFile::readRestartFile(_engine));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}