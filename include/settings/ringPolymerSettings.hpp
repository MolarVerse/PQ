#ifndef _RING_POLYMER_SETTINGS_HPP_

#define _RING_POLYMER_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class RingPolymerSettings
     *
     * @brief class for storing settings for ring polymer md
     *
     */
    class RingPolymerSettings
    {
      private:
        static inline bool _numberOfBeadsSet = false;

        static inline size_t _numberOfBeads = 0;

      public:
        static void setNumberOfBeads(const size_t numberOfBeads);

        [[nodiscard]] static size_t getNumberOfBeads() { return _numberOfBeads; }
        [[nodiscard]] static bool   isNumberOfBeadsSet() { return _numberOfBeadsSet; }
    };
}   // namespace settings

#endif   // _RING_POLYMER_SETTINGS_HPP_