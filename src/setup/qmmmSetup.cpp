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
void QMMMSetup::setup() { setupQMCenter(); }

void QMMMSetup::setupQMCenter()
{
    std::string restartFileName       = settings::FileSettings::getStartFileName();
    std::string moldescriptorFileName = settings::FileSettings::getMolDescriptorFileName();
    std::string qmCenterString        = settings::QMMMSettings::getQMCenterString();

#ifdef PYTHON_ENABLED
    std::vector<int> qmCenter =
        pq_python::select(qmCenterString, restartFileName, moldescriptorFileName);
#else
    // check if string contains any characters that are not digits or commas
    if (qmCenterString.find_first_not_of("0123456789,") != std::string::npos)
    {
        throw customException::InputFileException(std::format(
            "The qm_center string {} contains characters that are not digits or commas. The "
            "current build of PQ was compiled without Python bindings, so the qm_center string "
            "must be a comma-separated list of integers, representing the atom indices in the "
            "restart file that should be treated as the QM center."
            "In order to use the full selection parser power of the PQAnalysis Python package, "
            "the PQ build must be compiled with Python bindings.",
            qmCenterString
        ));
    }

    // parse the qm_center string
    std::vector<int> qmCenter;
    size_t           pos = 0;
    while (pos < qmCenterString.size())
    {
        size_t nextPos = qmCenterString.find(',', pos);
        if (nextPos == std::string::npos)
        {
            nextPos = qmCenterString.size();
        }
        std::string_view atomIndexString(qmCenterString.c_str() + pos, nextPos - pos);
        qmCenter.push_back(std::stoi(std::string(atomIndexString)));
        pos = nextPos + 1;
    }

    for (int i : qmCenter)
    {
        printf("%d\n", i);
    }

#endif
}