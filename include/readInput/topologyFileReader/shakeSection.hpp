#ifndef _SHAKE_SECTION_HPP_

#define _SHAKE_SECTION_HPP_

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
     * @class ShakeSection
     *
     * @brief reads shake section of topology file
     *
     */
    class ShakeSection : public TopologySection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "shake"; }
        void                      processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void                      endedNormally(bool) const override;
    };
}   // namespace readInput::topology

#endif   // _SHAKE_SECTION_HPP_