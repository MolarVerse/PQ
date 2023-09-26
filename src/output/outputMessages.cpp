#include "outputMessages.hpp"

#include "systemInfo.hpp"   // for _AUTHOR_

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
    std::stringstream header_title;

    header_title << R"(
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

    header_title << '\n';
    header_title << _OUTPUT_ << "Author: " << sysinfo::_AUTHOR_ << '\n';
    header_title << _OUTPUT_ << "Email:  " << sysinfo::_EMAIL_ << '\n';

    return header_title.str();
}

/**
 * @brief construct ended normally message
 *
 * @return string
 */
std::string output::endedNormally()
{
    const std::string endedNormally_message = R"(
*************************************************************************
*                                                                       *
*                      pimd-qmcf ended normally                         *
*                                                                       *
*************************************************************************
)";

    return endedNormally_message;
}

/**
 * @brief construct elapsed time message
 *
 * @param elapsedTime
 * @return string
 */
std::string output::elapsedTimeMessage(const double elapsedTime)
{
    return std::format("\n\n{}Elapsed time = {} s\n", _OUTPUT_, elapsedTime);
}