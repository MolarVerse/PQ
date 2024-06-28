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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>   // for std::string

#include "qmRunner.hpp"
#include "typeAliases.hpp"

namespace QM
{
    /**
     * @brief ASEQMRunner inherits from QMRunner
     *
     */
    class __attribute__((visibility("default"))) ASEQMRunner : public QMRunner
    {
       protected:
        double                    _energy;
        pybind11::object          _calculator;
        pybind11::object          _atoms_module;
        pybind11::array_t<double> _forces;
        pybind11::array_t<double> _stress_tensor;

       public:
        ASEQMRunner()           = default;
        ~ASEQMRunner() override = default;

        void         run(pq::SimBox &, pq::PhysicalData &) override;
        virtual void execute(pq::SimBox &)                         = 0;
        virtual void collectData(pq::SimBox &, pq::PhysicalData &) = 0;
    };
}   // namespace QM

#endif   // _ASE_QM_RUNNER_HPP_