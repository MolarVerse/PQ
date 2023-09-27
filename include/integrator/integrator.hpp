/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Atom;            // forward declaration
}   // namespace simulationBox

namespace integrator
{
    /**
     * @class Integrator
     *
     * @brief Integrator is a base class for all integrators
     *
     */
    class Integrator
    {
      protected:
        std::string _integratorType;

      public:
        Integrator() = default;
        explicit Integrator(const std::string_view integratorType) : _integratorType(integratorType){};
        virtual ~Integrator() = default;

        virtual void firstStep(simulationBox::SimulationBox &)  = 0;
        virtual void secondStep(simulationBox::SimulationBox &) = 0;

        void integrateVelocities(simulationBox::Atom *) const;
        void integratePositions(simulationBox::Atom *, const simulationBox::SimulationBox &) const;

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] std::string_view getIntegratorType() const { return _integratorType; }
    };

    /**
     * @class VelocityVerlet inherits Integrator
     *
     * @brief VelocityVerlet is a class for velocity verlet integrator
     *
     */
    class VelocityVerlet : public Integrator
    {
      public:
        explicit VelocityVerlet() : Integrator("VelocityVerlet"){};

        void firstStep(simulationBox::SimulationBox &) override;
        void secondStep(simulationBox::SimulationBox &) override;
    };

}   // namespace integrator

#endif   // _INTEGRATOR_HPP_