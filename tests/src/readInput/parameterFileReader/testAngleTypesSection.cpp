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
TEST_F(TestParameterFileSection, processSectionAngle)
{
    std::vector<std::string>               lineElements = {"0", "1.22", "234.3"};
    readInput::parameterFile::AngleSection angleSection;
    angleSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getAngleTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getAngleTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getAngleTypes()[0].getEquilibriumAngle(), 1.22);
    EXPECT_EQ(_engine->getForceField().getAngleTypes()[0].getForceConstant(), 234.3);

    lineElements = {"1", "2", "1.0", "0"};
    EXPECT_THROW(angleSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyAngle)
{
    auto angleSection = AngleSection();
    ASSERT_NO_THROW(angleSection.endedNormally(true));

    ASSERT_THROW_MSG(angleSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file angles section ended abnormally!");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}