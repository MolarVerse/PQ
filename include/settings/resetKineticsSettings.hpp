#ifndef _RESET_KINETICS_SETTINGS_HPP_

#define _RESET_KINETICS_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class ResetKineticsSettings
     *
     * @brief static class to store settings of reset kinetics
     *
     */
    class ResetKineticsSettings
    {
      private:
        static inline size_t _nScale = 0;
        static inline size_t _fScale = 0;
        static inline size_t _nReset = 0;
        static inline size_t _fReset = 0;

      public:
        ResetKineticsSettings()  = default;
        ~ResetKineticsSettings() = default;

        static void setNScale(const size_t nScale) { _nScale = nScale; }
        static void setFScale(const size_t fScale) { _fScale = fScale; }
        static void setNReset(const size_t nReset) { _nReset = nReset; }
        static void setFReset(const size_t fReset) { _fReset = fReset; }

        [[nodiscard]] static size_t getNScale() { return _nScale; }
        [[nodiscard]] static size_t getFScale() { return _fScale; }
        [[nodiscard]] static size_t getNReset() { return _nReset; }
        [[nodiscard]] static size_t getFReset() { return _fReset; }
    };
}   // namespace settings

#endif   // _RESET_KINETICS_SETTINGS_HPP_