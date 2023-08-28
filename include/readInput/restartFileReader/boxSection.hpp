#ifndef _BOX_SECTION_HPP_

#define _BOX_SECTION_HPP_

#include "restartFileSection.hpp"

#include <string>   // for string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::restartFile
{
    /**
     * @class BoxSection
     *
     * @brief Reads the box section of a .rst file
     *
     */
    class BoxSection : public RestartFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "box"; }
        [[nodiscard]] bool        isHeader() override { return true; }
        void                      process(std::vector<std::string> &lineElements, engine::Engine &) override;
    };

}   // namespace readInput::restartFile

#endif   // _BOX_SECTION_HPP_