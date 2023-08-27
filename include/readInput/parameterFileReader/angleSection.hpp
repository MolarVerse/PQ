#ifndef _ANGLE_SECTION_HPP_

#define _ANGLE_SECTION_HPP_

#include "parameterFileSection.hpp"

namespace readInput::parameterFile
{
    /**
     * @class AngleSection
     *
     * @brief reads angle section of parameter file
     *
     */
    class AngleSection : public ParameterFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "angles"; }

        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    };

}   // namespace readInput::parameterFile

#endif   // _ANGLE_SECTION_HPP_