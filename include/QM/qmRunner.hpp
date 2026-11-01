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

#ifndef _QM_RUNNER_HPP_

#define _QM_RUNNER_HPP_

#include <mstd/types.hpp>
#include <stop_token>

#include "box.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys
class QMSettings;   // forward declaration

namespace QM
{
    /**
     * @class QMRunner
     *
     * @brief base class for different qm engines
     *
     */
    class QMRunner
    {
       private:
        molsys::Periodicity _periodicity = molsys::Periodicity::NON_PERIODIC;
        mstd::ConstRef<QMSettings> _qmSettings;

       public:
        explicit QMRunner(const QMSettings &qmSettings);
        virtual ~QMRunner() = default;

        static void throwAfterTimeout(
            const std::stop_token &stopToken,
            double                 qmLoopTimeLimit
        );

        void         run(molsys::SimulationBox &, physicalData::PhysicalData &);
        virtual void run(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::Periodicity per
        ) = 0;

       protected:
        [[nodiscard]]
        molsys::Periodicity _getPeriodicity() const;

        void _setPeriodicity(molsys::Periodicity per);

        [[nodiscard]]
        const QMSettings &_getQMSettings() const;
    };
}   // namespace QM

#endif   // _QM_RUNNER_HPP_
