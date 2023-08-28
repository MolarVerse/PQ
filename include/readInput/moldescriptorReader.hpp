#ifndef _MOLDESCRIPTOR_READER_HPP_

#define _MOLDESCRIPTOR_READER_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace simulationBox
{
    class Molecule;   // Forward declaration
}

namespace readInput
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
        int               _lineNumber;
        const std::string _fileName;
        std::ifstream     _fp;

        engine::Engine &_engine;

      public:
        explicit MoldescriptorReader(engine::Engine &engine);

        void read();
        void processMolecule(std::vector<std::string> &lineElements);
        void convertExternalToInternalAtomTypes(simulationBox::Molecule &) const;
    };

}   // namespace readInput

#endif   // _MOLDESCRIPTOR_READER_HPP_