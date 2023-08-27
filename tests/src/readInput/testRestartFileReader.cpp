#include "testRestartFileReader.hpp"

#include "exceptions.hpp"
#include "moldescriptorReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace ::testing;
using namespace readInput;
using namespace customException;

/**
 * @brief tests determineSection base on the first element of the line
 *
 */
TEST_F(TestRstFileReader, determineSection)
{
    string                         filename = "examples/setup/h2o_qmcfc.rst";
    restartFile::RestartFileReader rstFileReader(filename, _engine);

    auto  lineElements = vector<string>{"sTeP", "1"};
    auto *section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "step");

    lineElements = vector<string>{"Box"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "box");

    lineElements = vector<string>{"notAHeaderSection"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "");
}

/**
 * @brief tests if the restart file is not found
 *
 */
TEST_F(TestRstFileReader, fileNotFound)
{
    string                         filename = "examples/setup/FILENOTFOUND.rst";
    restartFile::RestartFileReader rstFileReader(filename, _engine);

    ASSERT_THROW_MSG(rstFileReader.read(), InputFileException, "\"examples/setup/FILENOTFOUND.rst\" File not found");
}

/**
 * @brief test full read restart file function
 *
 */
TEST_F(TestRstFileReader, rstFileReading)
{
    _engine.getSettings().setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    MoldescriptorReader moldescriptor(_engine);

    string filename = "examples/setup/h2o-qmcf.rst";
    _engine.getSettings().setStartFilename(filename);

    moldescriptor.read();
    ASSERT_NO_THROW(restartFile::readRestartFile(_engine));
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}