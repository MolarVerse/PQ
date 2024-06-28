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

#include "maceRunner.hpp"

#include "pybind11/embed.h"

using QM::MaceRunner;

/**
 * @brief Construct a new MaceRunner::MaceRunner object
 *
 * @param modelType
 * @param model
 * @param fpType
 * @param dispersion
 *
 * @throw py::error_already_set if the import of the mace module fails
 */
MaceRunner::MaceRunner(
    const std::string &modelType,
    const std::string &model,
    const std::string &fpType,
    const bool         dispersion
)
    : ASEQMRunner()
{
    try
    {
        const py::module_ calculators = py::module_::import("mace.calculators");

        py::dict calculatorArgs;
        calculatorArgs["model"]         = model.c_str();
        calculatorArgs["dispersion"]    = pybind11::bool_(dispersion);
        calculatorArgs["default_dtype"] = fpType.c_str();
        calculatorArgs["device"]        = pybind11::str("cuda");

        _calculator = calculators.attr(modelType.c_str())(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}
