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

        // filenames and paths for later setup
        std::string _startFilename;
        std::string _moldescriptorFilename  = defaults::_MOLDESCRIPTOR_FILENAME_DEFAULT_;   // for backward compatibility
        std::string _guffDatFilename        = defaults::_GUFF_FILENAME_DEFAULT_;
        std::string _topologyFilename       = "";
        std::string _parameterFilename      = "";
        std::string _intraNonBondedFilename = "";

        std::string _jobtype;

        // noncoulomb settings for later setup
        std::string _nonCoulombType = defaults::_NONCOULOMB_TYPE_DEFAULT_;   // none

      public:
        /********************
         * standard getters *
         ********************/

        [[nodiscard]] std::string getStartFilename() const { return _startFilename; }
        [[nodiscard]] std::string getMoldescriptorFilename() const { return _moldescriptorFilename; }
        [[nodiscard]] std::string getGuffDatFilename() const { return _guffDatFilename; }
        [[nodiscard]] std::string getTopologyFilename() const { return _topologyFilename; }
        [[nodiscard]] std::string getParameterFilename() const { return _parameterFilename; }
        [[nodiscard]] std::string getIntraNonBondedFilename() const { return _intraNonBondedFilename; }

        [[nodiscard]] std::string getJobtype() const { return _jobtype; }

        [[nodiscard]] size_t getNScale() const { return _nScale; }
        [[nodiscard]] size_t getFScale() const { return _fScale; }
        [[nodiscard]] size_t getNReset() const { return _nReset; }
        [[nodiscard]] size_t getFReset() const { return _fReset; }

        [[nodiscard]] std::string getNonCoulombType() const { return _nonCoulombType; }

        /********************
         * standard setters *
         ********************/

        void setStartFilename(const std::string_view startFilename) { _startFilename = startFilename; }
        void setMoldescriptorFilename(const std::string_view filename) { _moldescriptorFilename = filename; }
        void setGuffDatFilename(const std::string_view guffDatFilename) { _guffDatFilename = guffDatFilename; }
        void setTopologyFilename(const std::string_view topologyFilename) { _topologyFilename = topologyFilename; }
        void setParameterFilename(const std::string_view parameterFilename) { _parameterFilename = parameterFilename; }
        void setIntraNonBondedFilename(const std::string_view filename) { _intraNonBondedFilename = filename; }

        void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }

        void setNScale(const size_t nScale) { _nScale = nScale; }
        void setFScale(const size_t fScale) { _fScale = fScale; }
        void setNReset(const size_t nReset) { _nReset = nReset; }
        void setFReset(const size_t fReset) { _fReset = fReset; }

        void setNonCoulombType(const std::string_view nonCoulombType) { _nonCoulombType = nonCoulombType; }
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_