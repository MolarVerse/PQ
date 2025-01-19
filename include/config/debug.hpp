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

#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <cstdlib>    // for size_t
#include <iostream>   // for operator<<, basic_ostream, ostream, cout

namespace config
{

    /**
     * @enum DebugLevel
     *
     * @brief Debug levels.
     *
     */
    enum class DebugLevel
    {
        ENERGY_DEBUG,
        FORCE_DEBUG,
        POSITION_DEBUG,
        VELOCITY_DEBUG,
    };

    class Debug
    {
       private:
        static bool inline _debug        = false;
        static size_t inline _debugLevel = 0;

        static size_t inline _functionLevel = 0;
        static int parseDebugBase(const char* debugEnvBase);
        static int parseDebugLevel(const char* debugEnv, const int debugBase);

        [[nodiscard]] static std::string getLeadingSpaces();
        [[nodiscard]] static std::string beautifyMsg(
            const std::string& msg,
            const char         beautifyChar
        );

       public:
        static void initDebug();

        template <typename T>
        static void debugMinMaxSumMean(
            std::tuple<T, T, T, T> minMaxSumMean,
            const std::string&     msg,
            const DebugLevel       level
        );

        [[nodiscard]] static bool useDebug(const DebugLevel level);
        [[nodiscard]] static bool useAnyDebug();

        static void enterFunction(const std::string& func);
        static void exitFunction(const std::string& func);
        static void debugInfo(const std::string& msg);

        static void setDebug(const bool debug);
        static void setDebugLevel(const size_t debugLevel);

        [[nodiscard]] static bool   getDebug();
        [[nodiscard]] static int    getDebugLevel();
        [[nodiscard]] static size_t getFunctionLevel();
    };

}   // namespace config

#ifndef __DEBUG_INL__
    #include "debug.inl"   // IWYU pragma: keep
#endif

#ifdef __PQ_DEBUG__

    #define __DEBUG_ENABLE__(call)         \
        do                                 \
        {                                  \
            if (config::Debug::getDebug()) \
            {                              \
                (call);                    \
            }                              \
        } while (0)

    #define __DEBUG_DISABLE__(call)         \
        do                                  \
        {                                   \
            if (!config::Debug::getDebug()) \
            {                               \
                (call);                     \
            }                               \
        } while (0)

    #define __DEBUG_ENTER_FUNCTION__(func)          \
        do                                          \
        {                                           \
            if (config::Debug::useAnyDebug())       \
            {                                       \
                config::Debug::enterFunction(func); \
            }                                       \
        } while (0)

    #define __DEBUG_EXIT_FUNCTION__(func)          \
        do                                         \
        {                                          \
            if (config::Debug::useAnyDebug())      \
            {                                      \
                config::Debug::exitFunction(func); \
            }                                      \
        } while (0)

    #define __DEBUG_INFO__(msg)                \
        do                                     \
        {                                      \
            if (config::Debug::useAnyDebug())  \
            {                                  \
                config::Debug::debugInfo(msg); \
            }                                  \
        } while (0)

#else

    #define __DEBUG_ENABLE__(call)           // Do nothing
    #define __DEBUG_DISABLE__(call)          // Do nothing
    #define __DEBUG_ENTER_FUNCTION__(func)   // Do nothing
    #define __DEBUG_EXIT_FUNCTION__(func)    // Do nothing
    #define __DEBUG_INFO__(msg)              // Do nothing

#endif

#endif   // __DEBUG_HPP__