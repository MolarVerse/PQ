#include "exceptions.hpp"
#include "parameterFileSection.hpp"
#include "testParameterFileSection.hpp"
#include "throwWithMessage.hpp"

using namespace ::testing;
using namespace readInput::parameterFile;

/**
 * @brief test bonds section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionDihedral)
{
    std::vector<std::string> lineElements = {"0", "1.22", "234.3", "324.3"};
    DihedralSection          dihedralSection;
    dihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getForceConstant(), 1.22);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getPeriodicity(), 234.3);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getPhaseShift(), 324.3);

    lineElements = {"1", "2", "1.0", "0", "2"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "-1.0", "3"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyDihedral)
{
    auto dihedralSection = DihedralSection();
    ASSERT_NO_THROW(dihedralSection.endedNormally(true));

    ASSERT_THROW_MSG(dihedralSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file dihedrals section ended abnormally!");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}