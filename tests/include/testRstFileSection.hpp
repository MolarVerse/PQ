#include "rstFileSection.hpp"
#include "engine.hpp"

using namespace Setup::RstFileReader;

class TestRstFileSection : public ::testing::Test
{
protected:
    virtual void SetUp() override
    {
        _section = new BoxSection;
        _engine = Engine();
    }

    virtual void TearDown() override
    {
        delete _section;
    }

    RstFileSection *_section;
    Engine _engine;
};