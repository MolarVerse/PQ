#ifndef _BOND_SECTION_HPP_

#define _BOND_SECTION_HPP_

#include "parameterFileSection.hpp"

#include <string>   // for allocator, string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::parameterFile
{
    /**
     * @class BondSection
     *
     * @brief reads bond section of parameter file
     *
     */
    class BondSection : public ParameterFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "bonds"; }

        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    };

}   // namespace readInput::parameterFile

#endif   // _BOND_SECTION_HPP_