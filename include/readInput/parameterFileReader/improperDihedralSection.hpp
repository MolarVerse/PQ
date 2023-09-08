#ifndef _IMPROPER_DIHEDRAL_SECTION_HPP_

#define _IMPROPER_DIHEDRAL_SECTION_HPP_

#include "parameterFileSection.hpp"   // for ParameterFileSection

#include <string>   // for allocator, string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::parameterFile
{
    /**
     * @class ImproperDihedralSection
     *
     * @brief reads improper dihedral section of parameter file
     *
     */
    class ImproperDihedralSection : public ParameterFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "impropers"; }

        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    };

}   // namespace readInput::parameterFile

#endif   // _IMPROPER_DIHEDRAL_SECTION_HPP_