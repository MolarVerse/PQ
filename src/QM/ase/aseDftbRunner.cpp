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

#include "aseDftbRunner.hpp"
#include "hubbardDerivMap.hpp"
#include "pybind11/embed.h"

using QM::AseDftbRunner;

/**
 * @brief Construct a new AseDftbRunner::AseDftbRunner object
 *
 * @param slakos
 *
 * @throw py::error_already_set if the import of the mace module fails
 */
AseDftbRunner::AseDftbRunner(
    const std::string &slakosType,
    const std::string &slakosPath
)
    : ASEQMRunner()
{
    try
    {
        const py::module_ calculator =
            py::module_::import("ase.calculators.dftb");

        const py::dict calculatorArgs;

        if (slakosType == "3ob" || slakosType == "matsci")
        {
            const std::string slakosDir = SLAKOS_DIR + slakosType + "/skfiles/";
            calculatorArgs["slako_dir"] = slakosDir.c_str();
        }
        else
            calculatorArgs["slako_dir"] = slakosPath.c_str();

        if (slakosType == "3ob")
        {
            setHubbDerivDict(constants::hubbardDerivMap3ob);
            calculatorArgs["Hamiltonian_ThirdOrderFull"] = "Yes";
            calculatorArgs["Hamiltonian_hubbardderivs_"] = "";
            const auto slakosDict                        = getHubbDerivDict();
            for (const auto &[key, value] : slakosDict)
            {
                auto _key = "Hamiltonian_hubbardderivs_" + key;
                calculatorArgs[_key.c_str()] = value;
            }
        }

        calculatorArgs["kpts"] = py::make_tuple(1, 1, 1);
        _calculator            = calculator.attr("Dftb")(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the 3ob Hubbard derivatives as dict
 *
 * @return std::unordered_map<std::string, float>
 */
const std::unordered_map<std::string, double> AseDftbRunner::getHubbDerivDict() const { return _slakosDict; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the 3ob Hubbard derivatives as dict
 */
void AseDftbRunner::setHubbDerivDict(
    const std::unordered_map<std::string, double> slakosDict
)
{
    _slakosDict = slakosDict;
}