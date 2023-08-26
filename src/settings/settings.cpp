#include "settings.hpp"

#include <utility>   // for make_pair

/**
 * @brief set relaxation time for manostat
 *
 * @param tauManostat
 *
 * @throw InputFileException if relaxation time is negative
 */
void settings::Settings::setTauManostat(double tauManostat) { _tauManostat = std::make_pair(true, tauManostat); }
