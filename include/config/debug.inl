#ifndef __DEBUG_INL__
#define __DEBUG_INL__

#include <stdlib.h>   // for getenv

#include <bitset>     // for bitset
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
        const char* debugEnv = std::getenv("PQ_DEBUG");

        if (debugEnv != nullptr)
        {
            try
            {
                const auto debugLevel = std::atoi(debugEnv);

                if (debugLevel < 0)
                {
                    const auto msg = std::format(
                        "Invalid debug level {}. Using default debug level 0.",
                        debugLevel
                    );
                    std::cout
                        << "Invalid debug level . Using default debug level 0."
                        << std::endl;
                    _debug      = false;
                    _debugLevel = 0;
                }
                else
                {
                    _debug      = debugLevel > 0;
                    _debugLevel = static_cast<size_t>(debugLevel);

                    const auto msg = std::format(
                        "Debugging is activated with debug level {}.",
                        debugLevel
                    );

                    std::cout << msg << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                const auto msg = std::format(
                    "Invalid debug level {}. Using default debug level 0.",
                    debugEnv
                );
                std::cout << msg << std::endl;
                _debug      = false;
                _debugLevel = 0;
            }
        }
        else
        {
            _debug      = false;
            _debugLevel = 0;
        }
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
     * @brief get if debug is enabled
     *
     * @return bool
     */
    [[nodiscard]] bool inline Debug::getDebug() { return _debug; }

    /**
     * @brief get the debug level
     *
     * @return int
     */
    [[nodiscard]] int inline Debug::getDebugLevel() { return _debugLevel; }

    /**
     * @brief get the null stream
     *
     * @return std::ostream&
     */
    [[nodiscard]] inline std::ostream& Debug::getNullStream()
    {
        return _nullStream;
    }

    /**
     * @brief check if the debug level is set
     *
     * @param level
     * @return bool
     */
    [[nodiscard]] bool inline Debug::useDebug(const DebugLevel level)
    {
        if (!_debug)
            return false;

        if (std::bitset<8>(_debugLevel)[static_cast<size_t>(level)] == 1)
            return true;

        return false;
    }

}   // namespace config

#endif   // __DEBUG_INL__