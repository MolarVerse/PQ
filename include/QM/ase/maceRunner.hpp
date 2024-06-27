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

#ifndef _MACE_RUNNER_HPP_

#define _MACE_RUNNER_HPP_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aseQMRunner.hpp"   // for InternalQMRunner

namespace QM
{
    /**
     * @brief MaceRunner inherits from InternalQMRunner
     *
     */
    class __attribute__((visibility("default"))) MaceRunner
        : public InternalQMRunner
    {
       private:
        double                    _energy;
        pybind11::object          _calculator;
        pybind11::object          _atoms_module;
        pybind11::array_t<double> _forces;
        pybind11::array_t<double> _stress_tensor;

       public:
        explicit MaceRunner(const std::string &, const std::string &);
        ~MaceRunner() override = default;

        void execute(pq::SimBox &) override;
        void collectData(pq::SimBox &, pq::PhysicalData &) override;
    };
}   // namespace QM

#endif   // _MACE_RUNNER_HPP_