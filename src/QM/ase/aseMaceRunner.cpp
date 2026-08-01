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

#include "aseMaceRunner.hpp"

#include <cstdio>

using QM::AseMaceRunner;

/**
 * @brief Construct a new AseMaceRunner::AseMaceRunner object
 *
 * @param modelType
 * @param model
 * @param fpType
 * @param dispersion
 * @param enableCueq
 *
 * @throw pybind11::error_already_set if the import of the mace module fails
 */
AseMaceRunner::AseMaceRunner(
    const std::string &modelType,
    const std::string &model,
    const std::string &fpType,
    const bool         dispersion,
    const bool         enableCueq
)
    : AseQMRunner()
{
    try
    {
        const pybind11::module_ calculators =
            pybind11::module_::import("mace.calculators");

        const pybind11::dict calculatorArgs;

        calculatorArgs["model"]         = model.c_str();
        calculatorArgs["dispersion"]    = pybind11::bool_(dispersion);
        calculatorArgs["enable_cueq"]   = pybind11::bool_(enableCueq);
        calculatorArgs["default_dtype"] = fpType.c_str();
        calculatorArgs["device"]        = pybind11::str("cuda");

        setAseCalculator(calculators.attr(modelType.c_str())(**calculatorArgs));
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        if (enableCueq)
            ::fprintf(
                stderr,
                "\nPQ: mace_mode = fast uses cuequivariance-accelerated MACE "
                "kernels. These need 'cuequivariance', 'cuequivariance-torch' "
                "and -- the piece pip does NOT pull in automatically -- the "
                "matching CUDA ops package 'cuequivariance-ops-torch-cuXX', "
                "where XX is the CUDA major your torch was built against (e.g. "
                "cu13 for a CUDA-13 torch build: "
                "'pip install cuequivariance-ops-torch-cu13'). Install that to "
                "use fast mode, or set mace_mode = accurate for the standard "
                "e3nn evaluation.\n"
            );
        throw;
    }
}
