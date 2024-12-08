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
        BOX_DEBUG = 0,
        LOCATION_DEBUG,
        ENERGY_DEBUG,
    };

    class nullbuf : public std::streambuf
    {
       public:
        [[nodiscard]] int_type overflow(int_type c) override
        {
            return traits_type::not_eof(c);
        }
    };

    class Debug
    {
       private:
        static bool inline _debug        = false;
        static size_t inline _debugLevel = 0;
        static nullbuf inline _nullBuf;
        static std::ostream inline _nullStream = std::ostream(&_nullBuf);

       public:
        static void initDebug();
        static void setDebug(const bool debug);
        static void setDebugLevel(const size_t debugLevel);

        [[nodiscard]] static bool          getDebug();
        [[nodiscard]] static int           getDebugLevel();
        [[nodiscard]] static std::ostream& getNullStream();

        [[nodiscard]] static bool useDebug(const DebugLevel level);
    };

}   // namespace config

#include "debug.inl"

#ifdef __PQ_DEBUG__

#define __DEBUG_LOCATION__()                                             \
    do {                                                                 \
        if (config::Debug::useDebug(config::DebugLevel::LOCATION_DEBUG)) \
        {                                                                \
            std::cout << std::endl;                                      \
            std::cout << "File:     " << __FILE__ << std::endl;          \
            std::cout << "Function: " << __FUNCTION__ << std::endl;      \
            std::cout << std::endl;                                      \
        }                                                                \
    } while (0)

#define __DEBUG_ENABLE__(call)         \
    do {                               \
        if (config::Debug::getDebug()) \
        {                              \
            (call);                    \
        }                              \
    } while (0)

#define __DEBUG_DISABLE__(call)         \
    do {                                \
        if (!config::Debug::getDebug()) \
        {                               \
            (call);                     \
        }                               \
    } while (0)

#else

#define __DEBUG_LOCATION__()      // Do nothing
#define __DEBUG_ENABLE__(call)    // Do nothing
#define __DEBUG_DISABLE__(call)   // Do nothing

#endif

#endif   // __DEBUG_HPP__