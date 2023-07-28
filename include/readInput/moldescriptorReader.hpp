#ifndef _MOLDESCRIPTOR_READER_H_

#define _MOLDESCRIPTOR_READER_H_

#include "engine.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace readInput
{
    class MoldescriptorReader;
    void readMolDescriptor(engine::Engine &);
}   // namespace readInput

/**
 * @class MoldescriptorReader
 *
 * @brief Reads a moldescriptor file
 *
 */
class readInput::MoldescriptorReader
{
  private:
    int               _lineNumber;
    const std::string _filename;
    std::ifstream     _fp;

    engine::Engine &_engine;

  public:
    explicit MoldescriptorReader(engine::Engine &engine);

    void read();
    void processMolecule(std::vector<std::string> &);
    void convertExternalToInternalAtomtypes(simulationBox::Molecule &) const;
};

#endif