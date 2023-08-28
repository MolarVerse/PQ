#include "testParameterFileReader.hpp"

#include "angleSection.hpp"              // for AngleSection
#include "bondSection.hpp"               // for BondSection
#include "dihedralSection.hpp"           // for DihedralSection
#include "exceptions.hpp"                // for InputFileException, ParameterFileException
#include "forceField.hpp"                // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "nonCoulombicsSection.hpp"      // for NonCoulombicsSection
#include "potential.hpp"                 // for Potential
#include "settings.hpp"                  // for Settings
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG
#include "typesSection.hpp"              // for TypesSection

#include "gtest/gtest.h"   // for Message, TestPartResult, AssertHelper, Test
#include <vector>          // for vector

using namespace ::testing;

/**
 * @brief tests isNeeded function
 *
 * @return true if forceField is enabled
 * @return false
 */
TEST_F(TestParameterFileReader, isNeeded)
{
    EXPECT_FALSE(_parameterFileReader->isNeeded());

    _engine->getForceField().activate();
    EXPECT_TRUE(_parameterFileReader->isNeeded());
}

/**
 * @brief tests determine section function
 *
 */
TEST_F(TestParameterFileReader, determineSection)
{
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 6);

    const auto *section = _parameterFileReader->determineSection({"types"});
    EXPECT_EQ(typeid(*section), typeid(readInput::parameterFile::TypesSection));

    const auto *section1 = _parameterFileReader->determineSection({"bonds"});
    EXPECT_EQ(typeid(*section1), typeid(readInput::parameterFile::BondSection));

    const auto *section2 = _parameterFileReader->determineSection({"angles"});
    EXPECT_EQ(typeid(*section2), typeid(readInput::parameterFile::AngleSection));

    const auto *section3 = _parameterFileReader->determineSection({"dihedrals"});
    EXPECT_EQ(typeid(*section3), typeid(readInput::parameterFile::DihedralSection));

    const auto *section4 = _parameterFileReader->determineSection({"impropers"});
    EXPECT_EQ(typeid(*section4), typeid(readInput::parameterFile::ImproperDihedralSection));

    const auto *section5 = _parameterFileReader->determineSection({"nonCoulombics"});
    EXPECT_EQ(typeid(*section5), typeid(readInput::parameterFile::NonCoulombicsSection));

    EXPECT_THROW_MSG([[maybe_unused]] const auto dummy = _parameterFileReader->determineSection({"notAValidSection"}),
                     customException::ParameterFileException,
                     "Unknown or already parsed keyword \"notAValidSection\" in parameter file");
}

/**
 * @brief tests delete section function
 *
 * @details if section is found, section should be deleted
 *
 */
TEST_F(TestParameterFileReader, deleteSection)
{
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 6);

    const auto *section = _parameterFileReader->determineSection({"types"});
    _parameterFileReader->deleteSection(section);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 5);

    const auto *section1 = _parameterFileReader->determineSection({"bonds"});
    _parameterFileReader->deleteSection(section1);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 4);

    const auto *section2 = _parameterFileReader->determineSection({"angles"});
    _parameterFileReader->deleteSection(section2);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 3);

    const auto *section3 = _parameterFileReader->determineSection({"dihedrals"});
    _parameterFileReader->deleteSection(section3);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 2);

    const auto *section4 = _parameterFileReader->determineSection({"impropers"});
    _parameterFileReader->deleteSection(section4);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 1);

    const auto *section5 = _parameterFileReader->determineSection({"nonCoulombics"});
    _parameterFileReader->deleteSection(section5);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 0);
}

/**
 * @brief tests read function
 *
 * @details if filename is empty, exception should be thrown
 *
 */
TEST_F(TestParameterFileReader, read_fileNameEmpty)
{
    _engine->getForceField().activate();
    _parameterFileReader->setFilename("");
    EXPECT_THROW_MSG(_parameterFileReader->read(),
                     customException::InputFileException,
                     "Parameter file needed for requested simulation setup");
}

/**
 * @brief tests read function
 *
 * @details if forceField is not activated, reading should be skipped
 *
 */
TEST_F(TestParameterFileReader, read_notNeeded)
{
    _engine->getForceField().deactivate();
    _parameterFileReader->setFilename("");
    EXPECT_NO_THROW(_parameterFileReader->read());
}

/**
 * @brief tests read function
 *
 * @details if file does not exists, exception should be thrown
 *
 */
TEST_F(TestParameterFileReader, read_fileDoesNotExists)
{
    _engine->getForceField().activate();
    _parameterFileReader->setFilename("doesNotExists");
    EXPECT_THROW_MSG(
        _parameterFileReader->read(), customException::InputFileException, "Parameter file \"doesNotExists\" File not found");
}

TEST_F(TestParameterFileReader, readParameterFile)
{
    _engine->getForceField().activate();
    _engine->getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    _engine->getSettings().setParameterFilename("data/parameterFileReader/param.param");
    EXPECT_NO_THROW(readInput::parameterFile::readParameterFile(*_engine));
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}