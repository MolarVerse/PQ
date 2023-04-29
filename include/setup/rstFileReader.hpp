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
    using namespace std;

    class RstFileReader
    {
    private:
        const string _filename;
        Settings &_settings;
        vector<RstFileSection *> _sections;

    public:
        RstFileReader(string, Settings &);
        ~RstFileReader();

        unique_ptr<SimulationBox> read();
        RstFileSection *determineSection(vector<string>);
    };
}

std::unique_ptr<SimulationBox> read_rst(std::string, Settings &);

#endif