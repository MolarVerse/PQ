/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "dftbplusRunner.hpp"     // for DFTBPlusRunner
#include "exceptions.hpp"         // for InputFileException
#include "qmRunner.hpp"           // for QMRunner
#include "qmSettings.hpp"         // for QMMethod, QMSettings
#include "qmSetup.hpp"            // for QMSetup, setupQM
#include "qmmdEngine.hpp"         // for QMMDEngine
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS
#include <string>          // for allocator, basic_string

TEST(TestQMSetup, setup)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("test");
    setupQM.setup();

    EXPECT_EQ(typeid(dynamic_cast<QM::DFTBPlusRunner &>(*engine.getQMRunner())), typeid(QM::DFTBPlusRunner));

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(setupQM.setup(),
                     customException::InputFileException,
                     "A qm based jobtype was requested but no external program via \"qm_prog\" provided");
}

TEST(TestQMSetup, setupQMFull)
{
    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("test");

    engine::QMMDEngine engine;
    EXPECT_NO_THROW(setup::setupQM(engine));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}