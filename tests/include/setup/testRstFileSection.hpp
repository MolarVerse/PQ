#include "rstFileSection.hpp"
#include "engine.hpp"

namespace Setup::RstFileReader
{
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
            _section = new BoxSection;
            _engine = Engine();
        }

        void TearDown() override
        {
            delete _section;
        }

        RstFileSection *_section;
        Engine _engine;
    };

    class TestNoseHooverSection : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // GTEST_SKIP();
            _section = new NoseHooverSection;
            _engine = Engine();
        }

        void TearDown() override
        {
            delete _section;
        }

        RstFileSection *_section;
        Engine _engine;
    };

    class TestStepCountSection : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            _section = new StepCountSection;
            _engine = Engine();
        }

        void TearDown() override
        {
            delete _section;
        }

        RstFileSection *_section;
        Engine _engine;
    };

    class TestAtomSection : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            _section = new AtomSection;
            _engine = Engine();
        }

        void TearDown() override
        {
            delete _section;
        }

        RstFileSection *_section;
        Engine _engine;
    };
}