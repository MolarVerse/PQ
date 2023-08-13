#include "exceptions.hpp"
#include "parameterFileSection.hpp"
#include "testParameterFileSection.hpp"
#include "throwWithMessage.hpp"

using namespace ::testing;
using namespace readInput::parameterFile;

/**
 * @brief test types section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionTypes)
{
    std::vector<std::string>               lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "0.99"};
    readInput::parameterFile::TypesSection typesSection;
    typesSection.process(lineElements, *_engine);
    EXPECT_EQ(_engine->getIntraNonBonded().getScale14Coulomb(), 0.23);
    EXPECT_EQ(_engine->getIntraNonBonded().getScale14VanDerWaals(), 0.99);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "1.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "1.23", "0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "-0.23", "0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "-0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyTypes)
{
    auto typesSection = TypesSection();
    ASSERT_NO_THROW(typesSection.endedNormally(true));

    ASSERT_THROW_MSG(typesSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file types section ended abnormally!");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}