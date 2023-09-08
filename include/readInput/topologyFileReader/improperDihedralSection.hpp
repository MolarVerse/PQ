#ifndef _IMPROPER_DIHEDRAL_SECTION_HPP_

#define _IMPROPER_DIHEDRAL_SECTION_HPP_

#include "topologySection.hpp"

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::topology
{
    /**
     * @class ImproperDihedralSection
     *
     * @brief reads improper dihedral section of topology file
     *
     */
    class ImproperDihedralSection : public TopologySection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "impropers"; }
        void                      processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void                      endedNormally(bool) const override;
    };
}   // namespace readInput::topology

#endif   // _IMPROPER_DIHEDRAL_SECTION_HPP_