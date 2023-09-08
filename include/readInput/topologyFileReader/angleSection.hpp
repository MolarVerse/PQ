#ifndef _ANGLE_SECTION_HPP_

#define _ANGLE_SECTION_HPP_

#include "topologySection.hpp"   // for TopologySection

#include <string>   // for allocator, string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::topology
{
    /**
     * @class AngleSection
     *
     * @brief reads angle section of topology file
     *
     */
    class AngleSection : public TopologySection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "angles"; }
        void                      processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void                      endedNormally(bool) const override;
    };
}   // namespace readInput::topology

#endif   // _ANGLE_SECTION_HPP_