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

#include "fairchemRunner.hpp"

#include "pybind11/embed.h"

using QM::FairchemRunner;

/**
 * @brief Construct a new FairchemRunner::FairchemRunner object
 *
 * @param modelType
 *
 * @throw py::error_already_set if the import of the fairchem module fails
 */
FairchemRunner::FairchemRunner(const std::string &modelType) : ASEQMRunner()
{
    try
    {
        const py::module_ calculators =
            py::module_::import("fairchem.core.common.relaxation.ase_utils");

        const py::dict calculatorArgs;

        if (modelType == "odac23")
            calculatorArgs["model_name"] = "odac23";
        else
            throw customException::UserInputException(
                std::format("Fairchem {} model not recognized", modelType)
            );

        calculatorArgs["local_cache"] = "/tmp/fairchem_checkpoints/";

        _calculator = calculators.attr("OCPCalculator")(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}
