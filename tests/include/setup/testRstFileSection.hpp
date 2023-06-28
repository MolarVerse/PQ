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
        _section = new setup::BoxSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    setup::RstFileSection *_section;
    engine::Engine         _engine;
};

class TestNoseHooverSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new setup::NoseHooverSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    setup::RstFileSection *_section;
    engine::Engine         _engine;
};

class TestStepCountSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new setup::StepCountSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    setup::RstFileSection *_section;
    engine::Engine         _engine;
};

class TestAtomSection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _section = new setup::AtomSection;
        _engine  = engine::Engine();
    }

    void TearDown() override { delete _section; }

    setup::RstFileSection *_section;
    engine::Engine         _engine;
};