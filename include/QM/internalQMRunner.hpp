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

#ifndef _INTERNAL_QM_RUNNER_HPP_

#define _INTERNAL_QM_RUNNER_HPP_

#include <string>   // for std::string

#include "qmRunner.hpp"
#include "typeAliases.hpp"

namespace QM
{
    /**
     * @brief InternalQMRunner inherits from QMRunner
     *
     */
    class InternalQMRunner : public QMRunner
    {
       public:
        InternalQMRunner()           = default;
        ~InternalQMRunner() override = default;

        void         run(pq::SimBox &, pq::PhysicalData &) override;
        virtual void execute(pq::SimBox &)                         = 0;
        virtual void collectData(pq::SimBox &, pq::PhysicalData &) = 0;
    };
}   // namespace QM

#endif   // _INTERNAL_QM_RUNNER_HPP_