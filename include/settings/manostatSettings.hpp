#ifndef _MANOSTAT_SETTINGS_HPP_

#define _MANOSTAT_SETTINGS_HPP_

#include "defaults.hpp"

#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace settings
{
    /**
     * @enum ManostatType
     *
     * @brief enum class to store the type of the manostat
     *
     */
    enum class ManostatType
    {
        NONE,
        BERENDSEN,
        STOCHASTIC_RESCALING
    };

    /**
     * @enum Isotropy
     *
     * @brief enum class to store the isotropy of the manostat
     *
     */
    enum class Isotropy
    {
        ISOTROPIC,
        SEMI_ISOTROPIC,
        ANISOTROPIC
    };

    [[nodiscard]] std::string string(const ManostatType &manostatType);

    /**
     * @class ManostatSettings
     *
     * @brief static class to store settings of the manostat
     *
     */
    class ManostatSettings
    {
      private:
        static inline ManostatType _manostatType = ManostatType::NONE;
        static inline Isotropy     _isotropy     = Isotropy::ISOTROPIC;

        static inline bool _isPressureSet = false;

        static inline double _targetPressure;   // no default value - has to be set by user
        static inline double _tauManostat     = defaults::_BERENDSEN_MANOSTAT_RELAXATION_TIME_;   // 1.0 ps
        static inline double _compressibility = defaults::_COMPRESSIBILITY_WATER_DEFAULT_;        // 4.5e-5 1/bar

        static inline std::vector<size_t> _xyIsotropicIndices;
        static inline size_t              _anisotropicIndex;

      public:
        ManostatSettings()  = default;
        ~ManostatSettings() = default;

        static void setManostatType(const std::string_view &manostatType);
        static void setManostatType(const ManostatType &manostatType) { _manostatType = manostatType; }

        static void setIsotropy(const std::string_view &isotropy){};   // TODO: implement
        static void setIsotropy(const Isotropy &isotropy) { _isotropy = isotropy; }

        /***************************
         * standard setter methods *
         ***************************/

        static void setPressureSet(const bool pressureSet) { _isPressureSet = pressureSet; }
        static void setTargetPressure(const double targetPressure) { _targetPressure = targetPressure; }
        static void setTauManostat(const double tauManostat) { _tauManostat = tauManostat; }
        static void setCompressibility(const double compressibility) { _compressibility = compressibility; }
        static void setXYIsotropicIndices(const std::vector<size_t> &indices) { _xyIsotropicIndices = indices; }
        static void setAnisotropicIndex(const std::vector<size_t> &indices) { _xyIsotropicIndices = indices; }
        static void setAnisotropicIndex(const size_t anisotropicIndex) { _anisotropicIndex = anisotropicIndex; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static ManostatType        getManostatType() { return _manostatType; }
        [[nodiscard]] static bool                isPressureSet() { return _isPressureSet; }
        [[nodiscard]] static double              getTargetPressure() { return _targetPressure; }
        [[nodiscard]] static double              getTauManostat() { return _tauManostat; }
        [[nodiscard]] static double              getCompressibility() { return _compressibility; }
        [[nodiscard]] static std::vector<size_t> getXYIsotropicIndices() { return _xyIsotropicIndices; }
        [[nodiscard]] static size_t              getAnisotropicIndex() { return _anisotropicIndex; }
    };

}   // namespace settings

#endif   // _MANOSTAT_SETTINGS_HPP_