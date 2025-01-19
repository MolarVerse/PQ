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

#include <cstdlib>   // for size_t
#include <string>    // for string

#include "linearAlgebra.hpp"   // for Vector3D

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
        TEMPERATURE_DEBUG,
        MOMENTUM_DEBUG,
        PRESSURE_DEBUG,
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
            const DebugLevel       level,
            const std::string&     unit = ""
        );

        template <typename T>
        static void debugValue(
            const T&           value,
            const std::string& msg,
            const DebugLevel   level,
            const std::string& unit = ""
        );

        template <typename T>
        static void debugValue3D(
            const linearAlgebra::Vector3D<T>& value,
            const std::string&                msg,
            const DebugLevel                  level,
            const std::string&                unit = ""
        );

        static void debugTensor3D(
            const linearAlgebra::tensor3D& tensor,
            const std::string&             msg,
            const DebugLevel               level,
            const std::string&             unit = ""
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

#ifdef __PQ_DEBUG__
    #define __DEBUG_TEMPERATURE__(temp)            \
        config::Debug::debugValue(                 \
            temp,                                  \
            "Temperature:",                        \
            config::DebugLevel::TEMPERATURE_DEBUG, \
            "K"                                    \
        );

    #define __DEBUG_MOMENTUM__(mom)             \
        config::Debug::debugValue3D(            \
            mom,                                \
            "Momentum:",                        \
            config::DebugLevel::MOMENTUM_DEBUG, \
            "amu*Angstrom/fs"                   \
        );

    #define __DEBUG_ANGULAR_MOMENTUM__(angMom)  \
        config::Debug::debugValue3D(            \
            angMom,                             \
            "Angular momentum:",                \
            config::DebugLevel::MOMENTUM_DEBUG, \
            "amu*Angstrom^2/fs"                 \
        );

    #define __DEBUG_ATOMIC_KINETIC_ENERGY__(kinEnergy) \
        config::Debug::debugTensor3D(                  \
            kinEnergy,                                 \
            "Atomic kinetic energy tensor:",           \
            config::DebugLevel::ENERGY_DEBUG,          \
            "kcal/mol"                                 \
        );

    #define __DEBUG_MOLECULAR_KINETIC_ENERGY__(kinEnergy) \
        config::Debug::debugTensor3D(                     \
            kinEnergy,                                    \
            "Molecular kinetic energy tensor:",           \
            config::DebugLevel::ENERGY_DEBUG,             \
            "kcal/mol"                                    \
        );

    #define __DEBUG_KINETIC_ENERGY__(kinEnergy) \
        config::Debug::debugValue(              \
            kinEnergy,                          \
            "Kinetic energy:",                  \
            config::DebugLevel::ENERGY_DEBUG,   \
            "kcal/mol"                          \
        );

#else

    #define __DEBUG_TEMPERATURE__(temp)                     // Do nothing
    #define __DEBUG_MOMENTUM__(mom)                         // Do nothing
    #define __DEBUG_ANGULAR_MOMENTUM__(angMom)              // Do nothing
    #define __DEBUG_ATOMIC_KINETIC_ENERGY__(kinEnergy)      // Do nothing
    #define __DEBUG_MOLECULAR_KINETIC_ENERGY__(kinEnergy)   // Do nothing
    #define __DEBUG_KINETIC_ENERGY__(kinEnergy)             // Do nothing

#endif

#endif   // __DEBUG_HPP__