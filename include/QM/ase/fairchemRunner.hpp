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

#ifndef _FAIRCHEM_RUNNER_HPP_

#define _FAIRCHEM_RUNNER_HPP_

#include "aseQMRunner.hpp"   // for InternalQMRunner

namespace QM
{
    /**
     * @brief FairchemRunner inherits from ASEQMRunner
     *
     */
    class __attribute__((visibility("default"))) FairchemRunner
        : public ASEQMRunner
    {
       public:
        ~FairchemRunner() override = default;

        explicit FairchemRunner(const std::string& modelType);
    };
}   // namespace QM

#endif   // _FAIRCHEM_RUNNER_HPP_