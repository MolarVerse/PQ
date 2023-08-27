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
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                              _engine;
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
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                              _engine;
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
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                              _engine;
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
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::restartFile::RestartFileSection *_section;
    engine::Engine                              _engine;
};