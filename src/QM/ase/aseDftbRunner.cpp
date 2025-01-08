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

#include "pybind11/embed.h"

using QM::AseDftbRunner;

/**
 * @brief Construct a new AseDftbRunner::AseDftbRunner object
 *
 * @param slakos
 *
 * @throw py::error_already_set if the import of the mace module fails
 */
AseDftbRunner::AseDftbRunner(const std::string &slakosType, const std::string &slakosPath) : ASEQMRunner()
{
    try
    {
        const py::module_ calculator =
            py::module_::import("ase.calculators.dftb.Dftb");

        const py::dict calculatorArgs;

        if (slakosType == "3ob" || slakosType == "matsci")
        {
            const std::string slakosDir = SLAKOS_DIR + slakosType + "/skfiles/";
            calculatorArgs["slakos"]    = slakosDir.c_str();
        }
        else
            calculatorArgs["slakos"] = slakosType.c_str();

        if (slakosType == "3ob")
        {
            calculatorArgs["Hamiltonian_ThirdOrderFull"] = "Yes";
            calculatorArgs["Hamiltonian_hubbardderivs_"] = "";
            // const auto slakosDict = get3obHubbDerivDict();
            // for (const auto &[key, value] : slakosDict)
            // {
            //     auto _key = "Hamiltonian_hubbardderivs_" + key;
            //     calculatorArgs[_key.c_str()] = value;
            // }
        }

        calculatorArgs["kpts"] = py::make_tuple(1, 1, 1);
        _calculator            = calculator(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

// const auto AseDftbRunner::get3obHubbDerivDict() const
// {
//     std::unordered_map<std::string, float> slakosDict;
//     slakosDict["C"]  = -0.1492;
//     slakosDict["N"]  = -0.1535;
//     slakosDict["O"]  = -0.1575;
//     slakosDict["H"]  = -0.1857;
//     slakosDict["S"]  = -0.11;
//     slakosDict["P"]  = -0.14;
//     slakosDict["F"]  = -0.1623;
//     slakosDict["Cl"] = -0.0697;
//     slakosDict["Br"] = -0.0573;
//     slakosDict["I"]  = -0.0433;
//     slakosDict["Zn"] = -0.03;
//     slakosDict["Mg"] = -0.02;
//     slakosDict["Ca"] = -0.0340;
//     slakosDict["K"]  = -0.0339;
//     slakosDict["Na"] = -0.0454;
//     return slakosDict;
// }