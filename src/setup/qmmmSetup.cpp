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

#include "qmmmSetup.hpp"

#include <cstddef>       // for size_t
#include <format>        // for format
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "engine.hpp"         // for QMMMMDEngine
#include "exceptions.hpp"     // for InputFileException
#include "fileSettings.hpp"   // for FileSettings
#include "qmmmSettings.hpp"   // for QMMMSettings
#include "qmmmmdEngine.hpp"   // for QMMMEngine

#ifdef PYTHON_ENABLED
#include "selection.hpp"   // for select
#endif

using setup::QMMMSetup;

/**
 * @brief wrapper to build QMMMSetup object and call setup
 *
 * @param engine
 */
void setup::setupQMMM(engine::QMMMMDEngine &engine)
{
    engine.getStdoutOutput().writeSetup("QMMM setup");
    engine.getLogOutput().writeSetup("QMMM setup");

    QMMMSetup qmmmSetup(engine);
    qmmmSetup.setup();
}

/**
 * @brief setup QMMM-MD
 *
 */
void QMMMSetup::setup()
{
    setupQMCenter();
    setupQMOnlyList();
    setupMMOnlyList();
    throw customException::UserInputException("Not implemented");
}

/**
 * @brief setup QM center
 *
 * @details This function determines the indices of the atoms that should be treated as the QM
 * center. The QM center is the region of the system that is treated with QM methods. All
 * atomIndices that are part of the QM center are added to the QM center list in the simulation box.
 *
 */
void QMMMSetup::setupQMCenter()
{
    const auto qmCenter = parseSelection(settings::QMMMSettings::getQMCenterString(), "qm_center");
    _engine.getSimulationBox().addQMCenterAtoms(qmCenter);
}

/**
 * @brief setup QM only list
 *
 */
void QMMMSetup::setupQMOnlyList()
{
    const auto qmOnlyList =
        parseSelection(settings::QMMMSettings::getQMOnlyListString(), "qm_only_list");
    _engine.getSimulationBox().setupQMOnlyAtoms(qmOnlyList);
}

/**
 * @brief setup MM only list
 *
 */
void QMMMSetup::setupMMOnlyList()
{
    const auto mmOnlyList =
        parseSelection(settings::QMMMSettings::getMMOnlyListString(), "mm_only_list");
    _engine.getSimulationBox().setupMMOnlyAtoms(mmOnlyList);
}

/**
 * @brief parse selection string
 *
 * @details This function parses a string that contains a selection of atoms. The selection can be
 * a list of atom indices or a selection string that is understood by the PQAnalysis Python package.
 * In order to use the full selection parser power of the PQAnalysis Python package, the PQ build
 * must be compiled with Python bindings. If the PQ build is compiled without Python bindings, the
 * selection string must be a comma-separated list of integers, representing the atom indices in the
 * restart file that should be treated as the selection. If the selection is empty, the function
 * returns a vector with a single element, 0.
 *
 * @param selection The selection string
 * @param key The key of the selection string
 *
 * @return std::vector<int> The selection vector
 *
 * @throws customException::InputFileException if the selection string contains characters that are
 * not digits or commas and the PQ build is compiled without Python bindings.
 */
std::vector<int> QMMMSetup::parseSelection(const std::string &selection, const std::string &key)
{
    std::string restartFileName       = settings::FileSettings::getStartFileName();
    std::string moldescriptorFileName = settings::FileSettings::getMolDescriptorFileName();

    if (selection.empty())
        return {0};

#ifdef PYTHON_ENABLED
    std::vector<int> selectionVector =
        pq_python::select(selection, restartFileName, moldescriptorFileName);
#else
    // check if string contains any characters that are not digits or commas
    if (selection.find_first_not_of("0123456789,") != std::string::npos)
    {
        throw customException::InputFileException(std::format(
            "The value of key {} - {} contains characters that are not digits or commas. The "
            "current build of PQ was compiled without Python bindings, so the {} string "
            "must be a comma-separated list of integers, representing the atom indices in the "
            "restart file that should be treated as the {}."
            "In order to use the full selection parser power of the PQAnalysis Python package, "
            "the PQ build must be compiled with Python bindings.",
            key,
            selection,
            key,
            key
        ));
    }

    // parse the qm_center string
    std::vector<int> selectionVector;
    size_t           pos = 0;
    while (pos < selection.size())
    {
        size_t nextPos = selection.find(',', pos);
        if (nextPos == std::string::npos)
        {
            nextPos = selection.size();
        }
        std::string_view atomIndexString(selection.c_str() + pos, nextPos - pos);
        selectionVector.push_back(std::stoi(std::string(atomIndexString)));
        pos = nextPos + 1;
    }

#endif

    return selectionVector;
}