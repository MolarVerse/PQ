#include "settings.hpp"

#include "resetKineticsSettings.hpp"

/**
 * @brief Finalize the settings by finalizing each individual setting component
 */
void Settings::finalize() { resetKinetics.finalize(); }
