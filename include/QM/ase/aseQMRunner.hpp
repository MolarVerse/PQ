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

#ifndef _ASE_QM_RUNNER_HPP_

#define _ASE_QM_RUNNER_HPP_

#include "qmRunner.hpp"

namespace pybind11
{
    class object;
}

namespace QM
{
    /**
     * @brief AseQMRunner inherits from QMRunner
     *
     */
    class AseQMRunner : public QMRunner
    {
       protected:
        double _energy;

        struct AseInterface;
        std::unique_ptr<AseInterface> _ase;

       public:
        AseQMRunner();
        ~AseQMRunner() override;

        void run(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physData,
            simulationBox::Periodicity    periodicity
        ) override;
        void buildAseAtoms(const simulationBox::SimulationBox &);
        void execute();

        void collectData(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &
        ) const;
        void collectForces(simulationBox::SimulationBox &) const;
        void collectEnergy(physicalData::PhysicalData &) const;
        void collectStress(
            const simulationBox::SimulationBox &,
            physicalData::PhysicalData &
        ) const;

       protected:
        void setAseCalculator(const pybind11::object &calculator);
    };

}   // namespace QM

#endif   // _ASE_QM_RUNNER_HPP_
