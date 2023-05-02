#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include <string>
#include <memory>
#include <vector>

#include "simulationBox.hpp"
#include "settings.hpp"
#include "rstFileSection.hpp"

namespace Setup::RstFileReader
{
    class RstFileReader
    {
    private:
        const std::string _filename;
        Settings &_settings;
        std::vector<RstFileSection *> _sections = {new BoxSection, new NoseHooverSection, new StepCountSection};

    public:
        RstFileReader(const std::string &, Settings &);
        ~RstFileReader();

        std::unique_ptr<SimulationBox> read();
        RstFileSection *determineSection(std::vector<std::string> &);
    };
}

std::unique_ptr<SimulationBox> read_rst(std::string, Settings &);

#endif