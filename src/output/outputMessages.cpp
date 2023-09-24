#include "outputMessages.hpp"

#include <format>    // for format
#include <sstream>   // for stringstream
#include <string>    // for string

/**
 * @brief construct header title
 *
 * @return string
 */
std::string output::header()
{
    const std::string header_title = R"(
**************************************************************************************************
*                                                                                                *
*              $$\                     $$\                                            $$$$$$\    *
*              \__|                    $$ |                                          $$  __$$\   *
*     $$$$$$\  $$\ $$$$$$\$$$$\   $$$$$$$ |         $$$$$$\  $$$$$$\$$$$\   $$$$$$$\ $$ /  \__|  *
*    $$  __$$\ $$ |$$  _$$  _$$\ $$  __$$ |$$$$$$\ $$  __$$\ $$  _$$  _$$\ $$  _____|$$$$\       *
*    $$ /  $$ |$$ |$$ / $$ / $$ |$$ /  $$ |\______|$$ /  $$ |$$ / $$ / $$ |$$ /      $$  _|      *
*    $$ |  $$ |$$ |$$ | $$ | $$ |$$ |  $$ |        $$ |  $$ |$$ | $$ | $$ |$$ |      $$ |        *
*    $$$$$$$  |$$ |$$ | $$ | $$ |\$$$$$$$ |        \$$$$$$$ |$$ | $$ | $$ |\$$$$$$$\ $$ |        *
*    $$  ____/ \__|\__| \__| \__| \_______|         \____$$ |\__| \__| \__| \_______|\__|        *
*    $$ |                                                $$ |                                    *
*    $$ |                                                $$ |                                    *
*    \__|                                                \__|                                    *
*                                                                                                *
*                                                                                                *
**************************************************************************************************
)";

    return header_title;
}

/**
 * @brief construct general initial momentum message
 *
 * @param momentum
 * @return string
 */
std::string output::initialMomentumMessage(const double initialMomentum)
{
    return std::format("Initial momentum = {} Angstrom * amu / fs", initialMomentum);
}