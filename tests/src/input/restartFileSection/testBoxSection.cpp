/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "box.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "gmock/gmock.h"
#include "manostatSettings.hpp"
#include "restartFileReader/restartFileSection.hpp"
#include "testRestartFileSection.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

TEST_F(TestBoxSection, testKeyword) { EXPECT_EQ(_section->keyword(), "box"); }

TEST_F(TestBoxSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestBoxSection, testNumberOfArguments)
{
    size_t lineNumber = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        if (i != 4 && i != 7)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW_MSG(
                _section->process(line, *_engine, lineNumber),
                exc::RstFileException,
                "Error in line 0: Box section must have 4 or 7 elements"
            );
        }
    }
}

TEST_F(TestBoxSection, testProcess)
{
    _settings.generalSettings.setJobtypeByUser(JobType::QM_MD);

    EXPECT_EQ(_wasBoxSetViaStartFile(), false);

    size_t lineNumber = 0;

    std::vector<std::string> line = {"box", "1.0", "2.0", "3.0"};
    _section->process(line, *_engine, lineNumber);
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxDimensions().numerical_value_in(
            angstrom
        ),
        testing::ElementsAre(1.0, 2.0, 3.0)
    );
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxAngles().numerical_value_in(deg),
        testing::ElementsAre(90.0, 90.0, 90.0)
    );

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "70.0"};
    _section->process(line, *_engine, lineNumber);
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxDimensions().numerical_value_in(
            angstrom
        ),
        testing::ElementsAre(1.0, 2.0, 3.0)
    );
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxAngles().numerical_value_in(deg),
        testing::ElementsAre(90.0, 90.0, 70.0)
    );

    line = {"box", "1.0", "2.0", "-3.0", "90.0", "90.0", "90.0"};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "All box dimensions must be positive"
    );

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "190.0"};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Box angles must be positive and smaller than 180°"
    );

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "-90.0"};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Box angles must be positive and smaller than 180°"
    );

    EXPECT_EQ(_wasBoxSetViaStartFile(), true);
}

TEST_F(TestBoxSection, testFullAnisotropicManostatNeedsABoxThatCanShear)
{
    _settings.generalSettings.setJobtypeByUser(JobType::QM_MD);
    _settings.manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings.manostatSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);

    size_t lineNumber = 0;

    std::vector<std::string> line = {"box", "1.0", "2.0", "3.0"};
    _section->process(line, *_engine, lineNumber);

    const auto &box = _engine->getSimulationBox().getBox();
    EXPECT_FALSE(box.isOrthorhombic());
    EXPECT_TRUE(box.allowChangeOfAngles());
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxDimensions().numerical_value_in(
            angstrom
        ),
        testing::ElementsAre(1.0, 2.0, 3.0)
    );
    ASSERT_THAT(
        _engine->getSimulationBox().getBoxAngles().numerical_value_in(deg),
        testing::ElementsAre(90.0, 90.0, 90.0)
    );
}

TEST_F(TestBoxSection, testOtherManostatSettingsKeepAnOrthorhombicBox)
{
    _settings.generalSettings.setJobtypeByUser(JobType::QM_MD);
    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"box", "1.0", "2.0", "3.0"};

    _settings.manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings.manostatSettings.setIsotropyByUser(Isotropy::ANISOTROPIC);
    _section->process(line, *_engine, lineNumber);
    EXPECT_TRUE(_engine->getSimulationBox().getBox().isOrthorhombic());

    _settings.manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    _settings.manostatSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);
    _section->process(line, *_engine, lineNumber);
    EXPECT_TRUE(_engine->getSimulationBox().getBox().isOrthorhombic());
}

TEST_F(TestBoxSection, testTriclinicBoxMayChangeItsAnglesOnlyForFullAnisotropy)
{
    _settings.generalSettings.setJobtypeByUser(JobType::QM_MD);
    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"box", "1.0", "2.0", "3.0"};
    line.insert(line.end(), {"75", "80", "70"});

    _section->process(line, *_engine, lineNumber);
    EXPECT_FALSE(_engine->getSimulationBox().getBox().isOrthorhombic());
    EXPECT_FALSE(_engine->getSimulationBox().getBox().allowChangeOfAngles());

    _settings.manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings.manostatSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);
    _section->process(line, *_engine, lineNumber);
    EXPECT_TRUE(_engine->getSimulationBox().getBox().allowChangeOfAngles());
}

TEST_F(TestBoxSection, testMMMDRefusesATriclinicBox)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);

    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"box", "1.0", "2.0", "3.0"};
    line.insert(line.end(), {"90", "90", "70"});

    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::InputFileException,
        "Triclinic box is only supported for QM-MD and RP-QM-MD"
    );
}

TEST_F(TestBoxSection, testMMMDRefusesFullAnisotropicManostatForAnOrthogonalBox)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings.manostatSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);

    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"box", "1.0", "2.0", "3.0"};

    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::InputFileException,
        "isotropy = full_anisotropic needs a triclinic box (it changes the box "
        "angles), which is only supported for QM-MD and RP-QM-MD"
    );
}
