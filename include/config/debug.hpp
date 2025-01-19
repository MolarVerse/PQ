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
        LOCATION_DEBUG = 0,
        ENERGY_DEBUG,
        FORCE_DEBUG,
        POSITION_DEBUG,
        VELOCITY_DEBUG,
        BOX_DEBUG,

    };

    class NullBuf : public std::streambuf
    {
       public:
        [[nodiscard]] int_type overflow(const int_type c) override
        {
            return traits_type::not_eof(c);
        }
    };

    class Debug
    {
       private:
        static bool inline _debug        = false;
        static size_t inline _debugLevel = 0;
        static NullBuf inline _nullBuf;
        static std::ostream inline _nullStream = std::ostream(&_nullBuf);

        static std::string inline _file       = "FILE:        ";
        static std::string inline _func       = "FUNCTION:    ";
        static std::string inline _debugInfo  = "DEBUG INFO:  ";
        static std::string inline _debugPos   = "DEBUG POS:   ";
        static std::string inline _debugVel   = "DEBUG VEL:   ";
        static std::string inline _debugForce = "DEBUG FORCE: ";

       public:
        static void initDebug();

        template <typename T>
        static void debugMinMaxSumMean(
            std::tuple<T, T, T, T> minMaxSumMean,
            const std::string&     msg,
            const DebugLevel       level
        );

        static void setDebug(const bool debug);
        static void setDebugLevel(const size_t debugLevel);

        [[nodiscard]] static bool          getDebug();
        [[nodiscard]] static int           getDebugLevel();
        [[nodiscard]] static std::ostream& getNullStream();
        [[nodiscard]] static std::string   getFile();
        [[nodiscard]] static std::string   getFunc();
        [[nodiscard]] static std::string   getDebugInfo();
        [[nodiscard]] static std::string   getDebugPos();
        [[nodiscard]] static std::string   getDebugVel();
        [[nodiscard]] static std::string   getDebugForce();

        [[nodiscard]] static bool useDebug(const DebugLevel level);
        [[nodiscard]] static bool useAnyDebug();
    };

}   // namespace config

#ifndef __DEBUG_INL__
    #include "debug.inl"   // IWYU pragma: keep
#endif

#ifdef __PQ_DEBUG__

    #define __DEBUG_LOCATION__()                                             \
        do                                                                   \
        {                                                                    \
            if (config::Debug::useDebug(config::DebugLevel::LOCATION_DEBUG)) \
            {                                                                \
                std::cout << std::endl;                                      \
                std::cout << config::Debug::getFile() << __FILE__            \
                          << std::endl;                                      \
                std::cout << config::Debug::getFunc() << __FUNCTION__        \
                          << std::endl;                                      \
                std::cout << std::endl;                                      \
            }                                                                \
        } while (0)

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

    #define __DEBUG_INFO__(msg)                                              \
        do                                                                   \
        {                                                                    \
            if (config::Debug::useDebug(config::DebugLevel::LOCATION_DEBUG)) \
            {                                                                \
                std::cout << config::Debug::getDebugInfo() << (msg)          \
                          << std::endl;                                      \
            }                                                                \
        } while (0)

#else

    #define __DEBUG_LOCATION__()      // Do nothing
    #define __DEBUG_ENABLE__(call)    // Do nothing
    #define __DEBUG_DISABLE__(call)   // Do nothing
    #define __DEBUG_INFO__(msg)       // Do nothing

#endif

#endif   // __DEBUG_HPP__