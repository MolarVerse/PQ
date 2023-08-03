#include "exceptions.hpp"
#include "parameterFileSection.hpp"
#include "testParameterFileSection.hpp"
#include "throwWithMessage.hpp"

using namespace ::testing;
using namespace readInput::parameterFile;

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