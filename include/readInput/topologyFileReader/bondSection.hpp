#ifndef _BOND_SECTION_HPP_

#define _BOND_SECTION_HPP_

#include "topologySection.hpp"

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::topology
{
    /**
     * @class BondSection
     *
     * @brief reads bond section of topology file
     *
     */
    class BondSection : public TopologySection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "bonds"; }
        void                      processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void                      endedNormally(bool) const override;
    };
}   // namespace readInput::topology

#endif   // _BOND_SECTION_HPP_