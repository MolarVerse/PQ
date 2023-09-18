#ifndef _MOLDESCRIPTOR_READER_HPP_

#define _MOLDESCRIPTOR_READER_HPP_

#include "defaults.hpp"

#include <fstream>   // for ifstream
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace simulationBox
{
    class MoleculeType;   // Forward declaration
}

namespace readInput::molDescriptor
{
    void readMolDescriptor(engine::Engine &);

    /**
     * @class MoldescriptorReader
     *
     * @brief Reads a moldescriptor file
     *
     */
    class MoldescriptorReader
    {
      private:
        int           _lineNumber;
        std::string   _fileName = defaults::_MOLDESCRIPTOR_FILENAME_DEFAULT_;
        std::ifstream _fp;

        engine::Engine &_engine;

      public:
        explicit MoldescriptorReader(engine::Engine &engine);

        void read();
        void processMolecule(std::vector<std::string> &lineElements);
        void convertExternalToInternalAtomTypes(simulationBox::MoleculeType &) const;
    };

}   // namespace readInput::molDescriptor

#endif   // _MOLDESCRIPTOR_READER_HPP_