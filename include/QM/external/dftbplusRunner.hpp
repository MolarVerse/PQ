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

#include "externalQMRunner.hpp"   // for ExternalQMRunner
#include "typeAliases.hpp"

namespace QM
{
    /**
     * @class DFTBPlusRunner
     *
     * @brief class for running DFTB+ inheriting from ExternalQMRunner
     *
     */
    class DFTBPlusRunner : public ExternalQMRunner
    {
       private:
        bool _isFirstExecution = true;
        bool _usePointCharges  = false;

       public:
        void writeCoordsFile(pq::SimBox &) override;
        void writePointChargeFile(pq::SimBox &) override;
        void execute(pq::SimBox &) override;
        void readStressTensor(pq::Box &box, pq::PhysicalData &data) override;
    };
}   // namespace QM

#endif   // _DFTBPLUS_RUNNER_HPP_