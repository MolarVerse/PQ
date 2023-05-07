#ifndef _MOLDESCRIPTOR_READER_H_

#define _MOLDESCRIPTOR_READER_H_

#include <string>
#include <vector>
#include <fstream>

#include "engine.hpp"

/**
 * @class MolDescriptorReader
 *
 * @brief Reads a moldescriptor file
 *
 */
class MolDescriptorReader
{
private:
    const std::string _filename;
    std::ifstream _fp;
    int _lineNumber;

public:
    explicit MolDescriptorReader(Engine &engine);
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