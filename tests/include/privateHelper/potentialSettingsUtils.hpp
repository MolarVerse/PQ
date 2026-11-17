#ifndef TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP
#define TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP

#include "potentialSettings.hpp"

struct PotentialSettingsUtils
{
    static void setEnergyCutoff(
        PotentialSettings& settings,
        double             energyCutoff
    )
    {
        settings._coulombConfig.energyCutoff = energyCutoff;
    }

    static void setForceCutoff(PotentialSettings& settings, double forceCutoff)
    {
        settings._coulombConfig.forceCutoff = forceCutoff;
    }
};

#endif   // TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP
