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

#include "atomSection.hpp"
#include "boxSection.hpp"
#include "engine.hpp"
#include "noseHooverSection.hpp"
#include "restartFileSection.hpp"
#include "stepCountSection.hpp"

#include <gtest/gtest.h>

/**
 * @class TestBoxSection
 *
 * @brief Test fixture for testing the BoxSection class.
 *
 */
class TestBoxSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new readInput::restartFile::BoxSection;
        _engine  = new engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                             *_engine;
};

/**
 * @class TestBondSection
 *
 * @brief Test fixture for testing the BondSection class.
 *
 */
class TestNoseHooverSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new readInput::restartFile::NoseHooverSection;
        _engine  = new engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                             *_engine;
};

/**
 * @class TestBondSection
 *
 * @brief Test fixture for testing the BondSection class.
 *
 */
class TestStepCountSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new readInput::restartFile::StepCountSection;
        _engine  = new engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                             *_engine;
};

/**
 * @class TestBondSection
 *
 * @brief Test fixture for testing the BondSection class.
 *
 */
class TestAtomSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new readInput::restartFile::AtomSection;
        _engine  = new engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                             *_engine;
};