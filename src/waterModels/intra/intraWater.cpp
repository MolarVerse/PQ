#include "intraWater.hpp"

namespace waterModel
{
    /**
     * @brief Construct a new Intra Water object
     *
     * @param hybridSettings The hybrid settings to be used
     */
    IntraWater::IntraWater(const HybridSettings& hybridSettings)
        : _hybridSettings(hybridSettings)
    {
    }

    /**
     * @brief Get the hybrid settings
     *
     * @return const HybridSettings&
     */
    const HybridSettings& IntraWater::_getHybridSettings() const
    {
        return _hybridSettings;
    }
}   // namespace waterModel
