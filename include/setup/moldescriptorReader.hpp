#ifndef _MOLDESCRIPTOR_READER_H_

#define _MOLDESCRIPTOR_READER_H_

#include <string>
#include <vector>
#include <fstream>

#include "engine.hpp"

/**
 * @class MoldescriptorReader
 *
 * @brief Reads a moldescriptor file
 *
 */
class MoldescriptorReader
{
private:
    const std::string _filename;
    std::ifstream _fp;
    int _lineNumber;

public:
    explicit MoldescriptorReader(Engine &engine);
    Engine &_engine;

    void read();
    void processMolecule(std::vector<std::string> &lineElements);
};

/**
 * @brief Reads a moldescriptor file
 *
 * @param engine
 */
void readMolDescriptor(Engine &);

#endif