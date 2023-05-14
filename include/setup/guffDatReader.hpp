#ifndef _GUFF_DAT_READER_H_

#define _GUFF_DAT_READER_H_

#include <string>

#include "engine.hpp"

/**
 * @class GuffDatReader
 *
 * @brief reads the guff.dat file
 *
 */
class GuffDatReader
{
private:
    std::string _filename = "guff.dat";
    int _lineNumber = 1;

    Engine &_engine;

public:
    explicit GuffDatReader(Engine &engine) : _engine(engine) {}

    void parseLine(std::vector<std::string> &);
    void read();
};

void readGuffDat();

#endif // _GUFF_DAT_READER_H_