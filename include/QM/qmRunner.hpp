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

#define SCRIPT_PATH_  _SCRIPT_PATH_
#define SINGULARITY_  _SINGULARITY_
#define STATIC_BUILD_ _STATIC_BUILD_

#include <stop_token>
#include <string>

#include "timer.hpp"
#include "typeAliases.hpp"

namespace QM
{
    /**
     * @class enum Periodicity
     */
    enum class Periodicity : size_t
    {
        NON_PERIODIC,
        X,
        Y,
        Z,
        XY,
        XZ,
        YZ,
        XYZ
    };

    /**
     * @class QMRunner
     *
     * @brief base class for different qm engines
     *
     */
    class QMRunner : public timings::Timer
    {
       protected:
        Periodicity _periodicity;

       public:
        virtual ~QMRunner() = default;

        void         throwAfterTimeout(const std::stop_token stopToken) const;
        void         run(pq::SimBox &, pq::PhysicalData &);
        virtual void run(pq::SimBox &, pq::PhysicalData &, Periodicity per) = 0;
    };
}   // namespace QM

#endif   // _QM_RUNNER_HPP_