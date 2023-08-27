#ifndef _DIHEDRAL_SECTION_HPP_

#define _DIHEDRAL_SECTION_HPP_

#include "parameterFileSection.hpp"

namespace readInput::parameterFile
{
    /**
     * @class DihedralSection
     *
     * @brief reads dihedral section of parameter file
     *
     */
    class DihedralSection : public ParameterFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "dihedrals"; }

        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    };

}   // namespace readInput::parameterFile

#endif   // _DIHEDRAL_SECTION_HPP_