#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

/*
 * for _COMPRESSIBILITY_WATER_DEFAULT_
 * for _COULOMB_LONG_RANGE_TYPE_DEFAULT_
 * for _GUFF_FILENAME_DEFAULT_
 * for _MOLDESCRIPTOR_FILENAME_DEFAULT_
 * for _NONCOULOMB_TYPE_DEFAULT_
 * for _RATTLE_MAX_ITER_DEFAULT_
 * for _RATTLE_TOLERANCE_DEFAULT_
 * for _SHAKE_MAX_ITER_DEFAULT_
 * for _SHAKE_TOLERANCE_DEFAULT_
 * for _WOLF_PARAMETER_DEFAULT_
 */
#include "defaults.hpp"

#include <cstddef>       // for size_t
#include <string>        // for string, allocator
#include <string_view>   // for basic_string_view, string_view

namespace settings
{
    /**
     * @class Settings
     *
     * @brief
     *  Stores the settings of the simulation
     *  Additionally it stores all information needed for later setup of the simulation
     *
     */
    class Settings
    {
      private:
        // resetKineticsSettings for later setup
        size_t _nScale = 0;
        size_t _fScale = 0;
        size_t _nReset = 0;
        size_t _fReset = 0;

        std::string _jobtype;

      public:
        /********************
         * standard getters *
         ********************/

        [[nodiscard]] std::string getJobtype() const { return _jobtype; }

        [[nodiscard]] size_t getNScale() const { return _nScale; }
        [[nodiscard]] size_t getFScale() const { return _fScale; }
        [[nodiscard]] size_t getNReset() const { return _nReset; }
        [[nodiscard]] size_t getFReset() const { return _fReset; }

        /********************
         * standard setters *
         ********************/

        void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }

        void setNScale(const size_t nScale) { _nScale = nScale; }
        void setFScale(const size_t fScale) { _fScale = fScale; }
        void setNReset(const size_t nReset) { _nReset = nReset; }
        void setFReset(const size_t fReset) { _fReset = fReset; }
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_