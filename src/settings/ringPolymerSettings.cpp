#include "ringPolymerSettings.hpp"

using settings::RingPolymerSettings;

/**
 * @brief set number of beads for ring polymer md
 *
 * @param numberOfBeads
 */
void RingPolymerSettings::setNumberOfBeads(const size_t numberOfBeads)
{
    _numberOfBeads    = numberOfBeads;
    _numberOfBeadsSet = true;
}