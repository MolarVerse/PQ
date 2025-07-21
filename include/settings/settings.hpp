/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

#include <cstdint>       // for uint_fast32_t
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _DIMENSIONALITY_DEFAULT_

namespace settings
{
    /**
     * @enum JobType
     *
     * @brief enum class to store the type of the job
     *
     */
    enum class JobType
    {
        MM_MD,
        QM_MD,
        QMMM_MD,
        RING_POLYMER_QM_MD,
        MM_OPT,
        NONE
    };

    /**
     * @enum FPType
     *
     * @brief enum class to store the floating point type
     *
     */
    enum class FPType
    {
        FLOAT,
        DOUBLE
    };

    [[nodiscard]] std::string string(const JobType jobtype);

    /**
     * @class Settings
     *
     * @brief Stores the general settings of the simulation
     *
     */
    class Settings
    {
       private:
        static inline JobType       _jobtype;
        static inline FPType        _floatingPointType = FPType::DOUBLE;
        static inline uint_fast32_t _randomSeed;
        static inline bool          _isRandomSeedset = false;

        static inline bool _useKokkos = false;

        static inline bool _isRingPolymerMDActivated = false;

        // clang-format off
        static inline size_t _dimensionality = defaults::_DIMENSIONALITY_DEFAULT_;
        // clang-format on

       public:
        Settings()  = default;
        ~Settings() = default;

        /***************************
         * standard setter methods *
         ***************************/

        static void setJobtype(const std::string_view jobtype);
        static void setJobtype(const JobType jobtype);

        static void setFloatingPointType(const std::string_view);
        static void setFloatingPointType(const FPType);

        static void setRandomSeed(const uint_fast32_t randomSeed);
        static void setIsRandomSeedSet(const bool isRandomSeedSet);

        static void setIsRingPolymerMDActivated(const bool isRingPolymerMD);
        static void setDimensionality(const size_t dimensionality);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static JobType getJobtype();

        [[nodiscard]] static FPType      getFloatingPointType();
        [[nodiscard]] static std::string getFloatingPointPybindString();

        [[nodiscard]] static uint_fast32_t getRandomSeed();
        [[nodiscard]] static bool          isRandomSeedSet();

        [[nodiscard]] static size_t getDimensionality();

        /******************************
         * standard is-active methods *
         ******************************/

        static void activateKokkos();
        static void activateRingPolymerMD();
        static void deactivateRingPolymerMD();

        [[nodiscard]] static bool isQMOnlyJobtype();
        [[nodiscard]] static bool isMMActivated();
        [[nodiscard]] static bool isQMActivated();
        [[nodiscard]] static bool isQMMMActivated();
        [[nodiscard]] static bool isQMOnlyActivated();
        [[nodiscard]] static bool isMMOnlyActivated();
        [[nodiscard]] static bool isRingPolymerMDActivated();
        [[nodiscard]] static bool isMDJobType();
        [[nodiscard]] static bool isOptJobType();
        [[nodiscard]] static bool useKokkos();
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_