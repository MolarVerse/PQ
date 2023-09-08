#ifndef _STEP_COUNT_SECTION_HPP_

#define _STEP_COUNT_SECTION_HPP_

#include "restartFileSection.hpp"   // for RestartFileSection

#include <string>   // for string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::restartFile
{
    /**
     * @class StepCountSection
     *
     * @brief Reads the step count section of a .rst file
     *
     */
    class StepCountSection : public RestartFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "step"; }
        [[nodiscard]] bool        isHeader() override { return true; }
        void                      process(std::vector<std::string> &lineElements, engine::Engine &) override;
    };

}   // namespace readInput::restartFile

#endif   // _STEP_COUNT_SECTION_HPP_