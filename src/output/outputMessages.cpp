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
*************************************************************************
*                                                                       *
*                            _                                    ___   *
*          _                ( )                                 /'___)  *
*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *
*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *
*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *
*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *
*  | |                                    | |                           *
*  (_)                                    (_)                           *
*                                                                       *
*                                                                       *
*************************************************************************
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