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

#ifndef _INTEGRATOR_HPP_

#define _INTEGRATOR_HPP_

#include <string>        // for string
#include <string_view>   // for string_view

#include "timer.hpp"   // for Timer
#include "typeAliases.hpp"

namespace integrator
{
    /**
     * @class Integrator
     *
     * @brief Integrator is a base class for all integrators
     *
     */
    class Integrator : public timings::Timer
    {
       protected:
        std::string _integratorType;   // TODO: make enum

       public:
        explicit Integrator(const std::string_view integratorType);
        Integrator()          = default;
        virtual ~Integrator() = default;

        virtual void firstStep(pq::SimBox &)  = 0;
        virtual void secondStep(pq::SimBox &) = 0;

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] std::string_view getIntegratorType() const;
    };

}   // namespace integrator

#endif   // _INTEGRATOR_HPP_