#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include <string>
#include <memory>

#include "simulationBox.hpp"
#include "settings.hpp"

class RstFileReader
{
private:
    const std::string _filename;
    Settings &_settings;

public:
    RstFileReader(std::string, Settings &);
    ~RstFileReader();

    std::unique_ptr<SimulationBox> read();
};

std::unique_ptr<SimulationBox> read_rst(std::string, Settings &);
Settings _settings;

#endif