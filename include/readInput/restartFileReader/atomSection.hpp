#ifndef _ATOM_SECTION_HPP_

#define _ATOM_SECTION_HPP_

#include "restartFileSection.hpp"   // for RestartFileSection

#include <gtest/gtest_prod.h>   // for FRIEND_TEST
#include <string>               // for string
#include <vector>               // for vector

class TestAtomSection_testProcessAtomLine_Test;   // Friend test class

namespace engine
{
    class Engine;   // Forward declaration
}

namespace simulationBox
{
    class Molecule;   // Forward declaration
}

namespace readInput::restartFile
{
    /**
     * @class AtomSection
     *
     * @brief Reads the atom section of a .rst file
     *
     */
    class AtomSection : public RestartFileSection
    {
      private:
        void processAtomLine(std::vector<std::string> &lineElements, simulationBox::Molecule &) const;
        void checkAtomLine(std::vector<std::string> &lineElements, const simulationBox::Molecule &);

        FRIEND_TEST(::TestAtomSection, testProcessAtomLine);

      public:
        [[nodiscard]] std::string keyword() override { return ""; }
        [[nodiscard]] bool        isHeader() override { return false; }
        void                      checkNumberOfLineArguments(std::vector<std::string> &lineElements) const;
        void                      process(std::vector<std::string> &lineElements, engine::Engine &) override;
    };

}   // namespace readInput::restartFile

#endif   // _ATOM_SECTION_HPP_
