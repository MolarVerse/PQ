/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef __DEBUG_INL__
#define __DEBUG_INL__

#include <bitset>     // for bitset
#include <cstdlib>    // for getenv
#include <format>     // for format
#include <iostream>   // for operator<<, basic_ostream, ostream, cout
#include <string>     // for string

#include "debug.hpp"

namespace config
{
    /**
     * @brief initialize debug level with the environment variable PQ_DEBUG
     *
     */
    void inline Debug::initDebug()
    {
        const char* debugEnvBase = std::getenv("PQ_DEBUG_BASE");
        const auto  debugBase    = parseDebugBase(debugEnvBase);

        const auto debugEnv = std::getenv("PQ_DEBUG");
        _debugLevel         = parseDebugLevel(debugEnv, debugBase);
        _debug              = _debugLevel > 0;
    }

    /**
     * @brief parse the debug base
     *
     * @param debugEnvBase
     * @return int
     */
    inline int Debug::parseDebugBase(const char* debugEnvBase)
    {
        auto debugBase = 10;

        if (debugEnvBase != nullptr)
        {
            const auto invalidBaseLambda = [&debugBase](const auto corruptBase)
            {
                const auto msg = std::format(
                    "Invalid debug base \"{}\". Using default "
                    "debug base \"{}\".",
                    corruptBase,
                    debugBase
                );

                std::cout << msg << std::endl;
            };

            try
            {
                const auto base = std::strtol(debugEnvBase, nullptr, 10);

                if (base != 10 && base != 2)
                    invalidBaseLambda(debugEnvBase);
                else
                    debugBase = base;
            }
            catch (const std::exception& e)
            {
                invalidBaseLambda(debugEnvBase);
            }
        }

        const auto msg = std::format(
            "Debugging is activated with debug base {}.",
            debugBase
        );

        std::cout << msg << std::endl;

        return debugBase;
    }

    /**
     * @brief parse the debug level
     *
     * @param debugEnv
     * @param debugBase
     * @return int
     */
    inline int Debug::parseDebugLevel(const char* debugEnv, const int debugBase)
    {
        auto debugLevel = 0;

        if (debugEnv != nullptr)
        {
            const auto invalidLevelLambda =
                [&debugLevel](const auto corruptLevel)
            {
                const auto msg = std::format(
                    "Invalid debug level \"{}\". Using default debug "
                    "level \"{}\".",
                    corruptLevel,
                    debugLevel
                );

                std::cout << msg << std::endl;
            };

            try
            {
                const auto level = std::strtol(debugEnv, nullptr, debugBase);

                if (level < 0)
                    invalidLevelLambda(level);
                else
                    debugLevel = level;
            }
            catch (const std::exception& e)
            {
                invalidLevelLambda(debugEnv);
            }
        }

        const auto msg = std::format(
            "Debugging is activated with debug level {}.",
            debugLevel
        );

        std::cout << msg << std::endl;

        return debugLevel;
    }

    /**
     * @brief print min, max, sum and mean of a tuple
     *
     * @tparam std::tuple<T, T, T, T>
     * @param msg
     * @param minMaxSumMean
     * @param level
     */
    template <typename T>
    void inline Debug::debugMinMaxSumMean(
        const std::tuple<T, T, T, T> minMaxSumMean,
        const std::string&           msg,
        const DebugLevel             level
    )
    {
        if (useDebug(level))
        {
            const auto min  = std::get<0>(minMaxSumMean);
            const auto max  = std::get<1>(minMaxSumMean);
            const auto sum  = std::get<2>(minMaxSumMean);
            const auto mean = std::get<3>(minMaxSumMean);

            const auto minString  = std::format("min:  {:6.5e}\n", min);
            const auto maxString  = std::format("max:  {:6.5e}\n", max);
            const auto sumString  = std::format("sum:  {:6.5e}\n", sum);
            const auto meanString = std::format("mean: {:6.5e}\n", mean);

            std::cout << getLeadingSpaces() << msg << std::endl;
            std::cout << getLeadingSpaces() << minString;
            std::cout << getLeadingSpaces() << maxString;
            std::cout << getLeadingSpaces() << sumString;
            std::cout << getLeadingSpaces() << meanString;
            std::cout << std::endl;
        }
    }

    /**
     * @brief check if the debug level is set
     *
     * @param level
     * @return bool
     */
    bool inline Debug::useDebug(const DebugLevel level)
    {
        if (!_debug)
            return false;

        if (std::bitset<8>(_debugLevel)[static_cast<size_t>(level)] == 1)
            return true;

        return false;
    }

    /**
     * @brief check if the debug level is not set
     *
     * @param level
     * @return bool
     */
    bool inline Debug::useAnyDebug()
    {
        if (!_debug)
            return false;

        if (_debugLevel != 0)
            return true;

        return false;
    }

    /**
     * @brief enter a function
     *
     * @param func
     */
    void inline Debug::enterFunction(const std::string& func)
    {
        std::cout << beautifyMsg(">>  " + func, '>') << std::endl;
        _functionLevel++;
    }

    /**
     * @brief exit a function
     *
     * @param func
     */
    void inline Debug::exitFunction(const std::string& func)
    {
        _functionLevel--;
        std::cout << beautifyMsg("<<  " + func, '<') << std::endl;
    }

    /**
     * @brief print debug info
     *
     * @param msg
     */
    void inline Debug::debugInfo(const std::string& msg)
    {
        std::cout << beautifyMsg(msg, '*') << std::endl;
    }

    /**
     * @brief beautify the message
     *
     * @param msg
     * @return std::string
     */
    std::string inline Debug::beautifyMsg(
        const std::string& msg,
        const char         beautifyChar
    )
    {
        const auto msgLen     = msg.length();
        const auto beautifier = std::string(msgLen, beautifyChar);

        auto beautifiedMsg  = getLeadingSpaces() + beautifier + "\n";
        beautifiedMsg      += getLeadingSpaces() + msg + "\n";
        beautifiedMsg      += getLeadingSpaces() + beautifier + "\n";

        return beautifiedMsg;
    }

    /**
     * @brief set if debug is enabled
     *
     * @param debug
     */
    void inline Debug::setDebug(const bool debug) { _debug = debug; }

    /**
     * @brief set the debug level
     *
     * @param debugLevel
     */
    void inline Debug::setDebugLevel(const size_t debugLevel)
    {
        _debugLevel = debugLevel;
    }

    /**
     * @brief get the leading spaces
     *
     */
    std::string inline Debug::getLeadingSpaces()
    {
        std::string leadingSpaces = "";

        for (size_t i = 0; i < _functionLevel; ++i)
            leadingSpaces += "    ";

        return leadingSpaces;
    }

    /**
     * @brief get if debug is enabled
     *
     * @return bool
     */
    bool inline Debug::getDebug() { return _debug; }

    /**
     * @brief get the debug level
     *
     * @return int
     */
    int inline Debug::getDebugLevel() { return _debugLevel; }

    /**
     * @brief get the function level
     *
     * @return int
     */
    inline size_t Debug::getFunctionLevel() { return _functionLevel; }

}   // namespace config

#endif   // __DEBUG_INL__