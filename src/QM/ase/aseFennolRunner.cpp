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

#include "aseFennolRunner.hpp"

using QM::AseFennolRunner;

/**
 * @brief Construct a new AseFennolRunner::AseFennolRunner object
 *
 * @param modelPath
 * @param gpuPreprocessing
 * @param useFloat64
 *
 * @throw pybind11::error_already_set if the import of the fennol module fails
 */
AseFennolRunner::AseFennolRunner(
    const std::string &modelPath,
    const bool         gpuPreprocessing,
    const bool         useFloat64
)
    : AseQMRunner()
{
    try
    {
        const pybind11::module_ calculators =
            pybind11::module_::import("fennol.ase");

        const pybind11::dict calculatorArgs;

        calculatorArgs["model"]             = modelPath.c_str();
        calculatorArgs["gpu_preprocessing"] = pybind11::bool_(gpuPreprocessing);
        calculatorArgs["use_float64"]       = pybind11::bool_(useFloat64);

        setAseCalculator(calculators.attr("FENNIXCalculator")(**calculatorArgs)
        );
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}
