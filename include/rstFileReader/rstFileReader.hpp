#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include <string>
#include <memory>

#include "simulationBox.hpp"

class RstFileReader
{
private:
    std::string _filename;

public:
    RstFileReader(std::string);
    ~RstFileReader();

    // void read(string filename);
};

std::unique_ptr<SimulationBox> read_rst(std::string filename);

#endif