#ifndef _NOSE_HOOVER_SECTION_HPP_

#define _NOSE_HOOVER_SECTION_HPP_

#include "restartFileSection.hpp"   // for RestartFileSection

#include <string>   // for string
#include <vector>   // for vector

namespace readInput::restartFile
{
    /**
     * @class NoseHooverSection
     *
     * @brief Reads the Nose-Hoover section of a .rst file
     *        TODO: This section is not yet implemented
     *
     */
    class NoseHooverSection : public RestartFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "chi"; }
        [[nodiscard]] bool        isHeader() override { return true; }
        void                      process(std::vector<std::string> &lineElements, engine::Engine &) override;
    };

}   // namespace readInput::restartFile

#endif   // _NOSE_HOOVER_SECTION_HPP_