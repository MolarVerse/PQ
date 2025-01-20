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
            "Debugging is activated with debug level {} or {}b.",
            debugLevel,
            std::bitset<16>(debugLevel).to_string()
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
     * @param unit
     */
    template <typename T>
    void inline Debug::debugMinMaxSumMean(
        const std::tuple<T, T, T, T> minMaxSumMean,
        const std::string&           msg,
        const DebugLevel             level,
        const std::string&           unit
    )
    {
        if (useDebug(level))
        {
            const auto min  = std::get<0>(minMaxSumMean);
            const auto max  = std::get<1>(minMaxSumMean);
            const auto sum  = std::get<2>(minMaxSumMean);
            const auto mean = std::get<3>(minMaxSumMean);

            const auto minStr  = std::format("min:  {:+7.5e} {}", min, unit);
            const auto maxStr  = std::format("max:  {:+7.5e} {}", max, unit);
            const auto sumStr  = std::format("sum:  {:+7.5e} {}", sum, unit);
            const auto meanStr = std::format("mean: {:+7.5e} {}", mean, unit);

            const auto msgLength = minStr.length();
            const auto delimiter = std::string(msgLength, '-');

            std::cout << getLeadingSpaces() << msg << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << minStr << std::endl;
            std::cout << getLeadingSpaces() << maxStr << std::endl;
            std::cout << getLeadingSpaces() << sumStr << std::endl;
            std::cout << getLeadingSpaces() << meanStr << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * @brief print a value
     *
     * @tparam T
     * @param value
     * @param msg
     * @param level
     * @param unit
     */
    template <typename T>
    void inline Debug::debugValue(
        const T&           value,
        const std::string& msg,
        const DebugLevel   level,
        const std::string& unit
    )
    {
        if (useDebug(level))
        {
            const auto msgStr = std::format("{} {:+7.5e} {}", msg, value, unit);

            const auto msgLength = msgStr.length();
            const auto delimiter = std::string(msgLength, '-');

            std::cout << getLeadingSpaces() << msgStr << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * @brief print a 3D value
     *
     * @tparam T
     * @param value
     * @param msg
     * @param level
     * @param unit
     */
    template <typename T>
    void inline Debug::debugValue3D(
        const linearAlgebra::Vector3D<T>& value,
        const std::string&                msg,
        const DebugLevel                  level,
        const std::string&                unit
    )
    {
        if (useDebug(level))
        {
            const auto _norm = linearAlgebra::norm(value);

            const auto x = std::format("x:    {:+7.5e} {}", value[0], unit);
            const auto y = std::format("y:    {:+7.5e} {}", value[1], unit);
            const auto z = std::format("z:    {:+7.5e} {}", value[2], unit);

            const auto componentLength = x.length();
            const auto delimiter       = std::string(componentLength, '-');

            const auto norm = std::format("norm: {:+7.5e} {}", _norm, unit);

            std::cout << getLeadingSpaces() << msg << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << x << std::endl;
            std::cout << getLeadingSpaces() << y << std::endl;
            std::cout << getLeadingSpaces() << z << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << norm << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * @brief print a 3D tensor
     *
     * @param tensor
     * @param msg
     * @param level
     * @param unit
     */
    void inline Debug::debugTensor3D(
        const linearAlgebra::tensor3D& tensor,
        const std::string&             msg,
        const DebugLevel               level,
        const std::string&             unit
    )
    {
        if (useDebug(level))
        {
            std::cout << getLeadingSpaces() << msg << " in " << unit
                      << std::endl;

            const auto line1 = std::format(
                "| {:+7.5e} | {:+7.5e} | {:+7.5e} |",
                tensor[0][0],
                tensor[0][1],
                tensor[0][2]
            );

            const auto line2 = std::format(
                "| {:+7.5e} | {:+7.5e} | {:+7.5e} |",
                tensor[1][0],
                tensor[1][1],
                tensor[1][2]
            );

            const auto line3 = std::format(
                "| {:+7.5e} | {:+7.5e} | {:+7.5e} |",
                tensor[2][0],
                tensor[2][1],
                tensor[2][2]
            );

            const auto lineLength = line1.length();
            const auto delimiter  = std::string(lineLength, '-');

            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << line1 << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << line2 << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
            std::cout << getLeadingSpaces() << line3 << std::endl;
            std::cout << getLeadingSpaces() << delimiter << std::endl;
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

        if (std::bitset<16>(_debugLevel)[static_cast<size_t>(level)] == 1)
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