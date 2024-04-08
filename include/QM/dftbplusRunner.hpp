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

#ifndef _DFTBPLUS_RUNNER_HPP_

#define _DFTBPLUS_RUNNER_HPP_

#include "qmRunner.hpp"   // for QMRunner

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Box;             // forward declaration

}   // namespace simulationBox

namespace QM
{
    /**
     * @class DFTBPlusRunner
     *
     * @brief class for running DFTB+ inheriting from QMRunner
     *
     */
    class DFTBPlusRunner : public QMRunner
    {
      private:
        bool _isFirstExecution = true;

      public:
        void writeCoordsFile(simulationBox::SimulationBox &) override;
        void execute() override;
        void readStressTensor(simulationBox::Box &box, physicalData::PhysicalData &data) override;
    };
}   // namespace QM

#endif   // _DFTBPLUS_RUNNER_HPP_