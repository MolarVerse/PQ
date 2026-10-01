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

#ifndef _TEST_RESTART_FILE_SECTION_HPP_
#define _TEST_RESTART_FILE_SECTION_HPP_

#include <gtest/gtest.h>

#include "mmmdEngine.hpp"
#include "restartFileReader/atomSection.hpp"
#include "restartFileReader/boxSection.hpp"
#include "restartFileReader/noseHooverSection.hpp"
#include "restartFileReader/restartFileSection.hpp"
#include "restartFileReader/stepCountSection.hpp"

/**
 * @class TestBoxSection
 *
 * @brief Test fixture for testing the BoxSection class.
 *
 */
class TestBoxSection : public ::testing::Test
{
   protected:
    std::unique_ptr<input::restartFile::RestartFileSection> _section;
    std::unique_ptr<engine::Engine>                         _engine;

    void SetUp() override
    {
        _section = std::make_unique<input::restartFile::BoxSection>();

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = std::make_unique<engine::MMMDEngine>();
    }
};

/**
 * @class TestNoseHooverSection
 *
 * @brief Test fixture for testing the NoseHooverSection class.
 *
 */
class TestNoseHooverSection : public ::testing::Test
{
   protected:
    std::unique_ptr<input::restartFile::RestartFileSection> _section;
    std::unique_ptr<engine::Engine>                         _engine;

    void SetUp() override
    {
        _section = std::make_unique<input::restartFile::NoseHooverSection>();

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = std::make_unique<engine::MMMDEngine>();
    }
};

/**
 * @class TestStepCountSection
 *
 * @brief Test fixture for testing the StepCountSection class.
 *
 */
class TestStepCountSection : public ::testing::Test
{
   protected:
    std::unique_ptr<input::restartFile::RestartFileSection> _section;
    std::unique_ptr<engine::Engine>                         _engine;

    void SetUp() override
    {
        _section = std::make_unique<input::restartFile::StepCountSection>();

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = std::make_unique<engine::MMMDEngine>();
    }
};

/**
 * @class TestAtomSection
 *
 * @brief Test fixture for testing the AtomSection class.
 *
 */
class TestAtomSection : public ::testing::Test
{
   protected:
    std::unique_ptr<input::restartFile::RestartFileSection> _section;
    std::unique_ptr<engine::Engine>                         _engine;

    static void _processAtomLine(
        std::vector<std::string> &line,
        molsys::SimulationBox    &simulationBox,
        molsys::Molecule         &molecule
    )
    {
        input::restartFile::AtomSection::_processAtomLine(
            line,
            simulationBox,
            molecule
        );
    }

    static void _processQMAtomLine(
        std::vector<std::string> &line,
        molsys::SimulationBox    &simulationBox
    )
    {
        input::restartFile::AtomSection::_processQMAtomLine(
            line,
            simulationBox
        );
    }

    void SetUp() override
    {
        _section = std::make_unique<input::restartFile::AtomSection>();

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = std::make_unique<engine::MMMDEngine>();
    }
};

#endif   // _TEST_RESTART_FILE_SECTION_HPP_
