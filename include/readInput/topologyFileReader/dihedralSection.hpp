#ifndef _DIHEDRAL_SECTION_HPP_

#define _DIHEDRAL_SECTION_HPP_

#include "topologySection.hpp"

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::topology
{
    /**
     * @class DihedralSection
     *
     * @brief reads dihedral section of topology file
     *
     */
    class DihedralSection : public TopologySection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "dihedrals"; }
        void                      processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void                      endedNormally(bool) const override;
    };
}   // namespace readInput::topology

#endif   // _DIHEDRAL_SECTION_HPP_