#ifndef _TYPES_SECTION_HPP_

#define _TYPES_SECTION_HPP_

#include "parameterFileSection.hpp"

#include <string>   // for allocator, string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::parameterFile
{
    /**
     * @class TypesSection
     *
     * @brief reads types line section of parameter file
     *
     */
    class TypesSection : public ParameterFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "types"; }

        void process(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    };

}   // namespace readInput::parameterFile

#endif   // _TYPES_SECTION_HPP_