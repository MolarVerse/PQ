#ifndef _MOLDESCRIPTOR_READER_HPP_

#define _MOLDESCRIPTOR_READER_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace simulationBox
{
    class Molecule;
}   // namespace simulationBox

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
        const std::string _filename;
        std::ifstream     _fp;

        engine::Engine &_engine;

      public:
        explicit MoldescriptorReader(engine::Engine &engine);

        void read();
        void processMolecule(std::vector<std::string> &);
        void convertExternalToInternalAtomTypes(simulationBox::Molecule &) const;
    };

}   // namespace readInput

#endif   // _MOLDESCRIPTOR_READER_HPP_