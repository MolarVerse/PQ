#ifndef _SETTINGS_HPP_
#define _SETTINGS_HPP_

#include "resetKineticsSettings.hpp"

/**
 * @brief Settings class that holds various simulation settings
 */
struct Settings
{
    ResetKineticsSettings resetKinetics;

    void finalize();
};

#endif   // _SETTINGS_HPP_
