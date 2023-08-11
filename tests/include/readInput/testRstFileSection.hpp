#include "engine.hpp"
#include "rstFileSection.hpp"

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
        _section = new readInput::BoxSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::RstFileSection *_section;
    engine::Engine             _engine;
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
        _section = new readInput::NoseHooverSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::RstFileSection *_section;
    engine::Engine             _engine;
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
        _section = new readInput::StepCountSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::RstFileSection *_section;
    engine::Engine             _engine;
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
        _section = new readInput::AtomSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    readInput::RstFileSection *_section;
    engine::Engine             _engine;
};