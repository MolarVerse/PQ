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

#ifndef _OPTIMIZER_SETTINGS_HPP_

#define _OPTIMIZER_SETTINGS_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string

#include "defaults.hpp"   // for _OPTIMIZER_DEFAULT_

namespace settings
{
    enum class Optimizer : size_t
    {
        NONE,
        GRADIENT_DESCENT,
    };

    std::string string(const Optimizer method);

    /**
     * @class OptimizerSettings
     *
     * @brief stores all information about the optimizer
     *
     */
    class OptimizerSettings
    {
       private:
        static inline Optimizer _optimizer = Optimizer::GRADIENT_DESCENT;

       public:
        static void setOptimizer(const std::string_view &optimizer);
        static void setOptimizer(const Optimizer optimizer);

        static std::string getOptimizer();
    };   // namespace settings
}   // namespace settings

#endif   // _OPTIMIZER_SETTINGS_HPP_
