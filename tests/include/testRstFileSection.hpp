#include "rstFileSection.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

using namespace Setup::RstFileReader;

class TestRstFileSection : public ::testing::Test
{
protected:
    virtual void SetUp() override
    {
        _section = new BoxSection;
        _settings = Settings();
        _simulationBox = SimulationBox();
    }

    virtual void TearDown() override
    {
        delete _section;
    }

    RstFileSection *_section;
    Settings _settings;
    SimulationBox _simulationBox;

};