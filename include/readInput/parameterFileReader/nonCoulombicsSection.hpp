#ifndef _NON_COULOMBICS_SECTION_HPP_

#define _NON_COULOMBICS_SECTION_HPP_

#include "parameterFileSection.hpp"   // for ParameterFileSection

#include <cstddef>   // for size_t
#include <string>    // for allocator, string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace potential
{
    enum class NonCoulombType : size_t;   // forward declaration
}

namespace readInput::parameterFile
{
    /**
     * @class NonCoulombicsSection
     *
     * @brief reads non-coulombics section of parameter file
     *
     */
    class NonCoulombicsSection : public ParameterFileSection
    {
      private:
        potential::NonCoulombType _nonCoulombType;

      public:
        [[nodiscard]] std::string keyword() override { return "noncoulombics"; }

        void processSection(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processHeader(std::vector<std::string> &lineElements, engine::Engine &) override;
        void processLJ(std::vector<std::string> &lineElements, engine::Engine &) const;
        void processBuckingham(std::vector<std::string> &lineElements, engine::Engine &) const;
        void processMorse(std::vector<std::string> &lineElements, engine::Engine &) const;

        void setNonCoulombType(potential::NonCoulombType nonCoulombicType) { _nonCoulombType = nonCoulombicType; }
        [[nodiscard]] potential::NonCoulombType getNonCoulombType() const { return _nonCoulombType; }
    };

}   // namespace readInput::parameterFile

#endif   // _NON_COULOMBICS_SECTION_HPP_