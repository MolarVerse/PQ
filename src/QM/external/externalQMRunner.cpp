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

#include "externalQMRunner.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <chrono>       // for seconds
#include <format>       // for format
#include <fstream>      // for ofstream
#include <functional>   // for identity
#include <string>       // for string
#include <thread>       // for sleep_for
#include <vector>       // for vector

#include "constants/conversionFactors.hpp"   // for _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_, _HARTREE_TO_KCAL_PER_MOL_
#include "exceptions.hpp"                    // for InputFileException
#include "physicalData.hpp"                  // for PhysicalData
#include "qmSettings.hpp"                    // for QMSettings
#include "simulationBox.hpp"                 // for SimulationBox
#include "vector3d.hpp"                      // for Vec3D

using QM::ExternalQMRunner;
using namespace simulationBox;
using namespace physicalData;
using namespace customException;
using namespace settings;
using namespace constants;





/**
 * @brief run the qm engine with a specific selection of molecules
 * 
 * @param molecules
 * @param simBox
 */
void ExternalQMRunner::run(const std::vector<Molecule> molecules, SimulationBox &simBox, PhysicalData &physicalData)
{

    
    auto partitionBox = simBox.selectPartitionBox(molecules);

    writeCoordsFile(*partitionBox);

    std::jthread timeoutThread{[this](const std::stop_token stopToken)
                               { throwAfterTimeout(stopToken); }};

    execute();

    timeoutThread.request_stop();

    readForceFile(*partitionBox, physicalData);
    readStressTensor(partitionBox->getBox(), physicalData);
}


/**
 * @brief run the qm engine with all molecules
 *
 * @param simBox
 */
void ExternalQMRunner::run(SimulationBox &simBox, PhysicalData &physicalData)
{   

    auto molecules = simBox.getMolecules();
    run(molecules, simBox, physicalData);
}


/**
 * @brief reads the force file (including qm energy) and sets the forces of the
 * atoms
 *
 * @param box
 * @param physicalData
 *
 * @throw QMRunnerException
 *  - if the force file cannot be opened
 *  - if the force file is empty
 */
void ExternalQMRunner::readForceFile(
    SimulationBox &box,
    PhysicalData  &physicalData
)
{
    const std::string forceFileName = "qm_forces";

    std::ifstream forceFile(forceFileName);

    if (!forceFile.is_open())
        throw QMRunnerException(std::format(
            "Cannot open {} force file \"{}\"",
            string(QMSettings::getQMMethod()),
            forceFileName
        ));

    if (forceFile.peek() == std::ifstream::traits_type::eof())
        throw QMRunnerException(std::format(
            "Empty {} force file \"{}\"",
            string(QMSettings::getQMMethod()),
            forceFileName
        ));

    double energy = 0.0;

    forceFile >> energy;

    physicalData.setQMEnergy(energy * _HARTREE_TO_KCAL_PER_MOL_);

    auto readForces = [&forceFile](auto &atom)
    {
        auto grad = linearAlgebra::Vec3D();

        forceFile >> grad[0] >> grad[1] >> grad[2];

        atom->setForce(-grad * _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_);
    };

    std::ranges::for_each(box.getQMAtoms(), readForces);

    forceFile.close();

    ::system(std::format("rm -f {}", forceFileName).c_str());
}

/*******************************
 *                             *
 * standard getter and setters *
 *                             *
 *******************************/

/**
 * @brief getter for the script path
 *
 * @return const std::string&
 */
const std::string &ExternalQMRunner::getScriptPath() const
{
    return _scriptPath;
}

/**
 * @brief getter for the singularity path
 *
 * @return const std::string&
 */
const std::string &ExternalQMRunner::getSingularity() const
{
    return _singularity;
}

/**
 * @brief getter for the static build path
 *
 * @return const std::string&
 */
const std::string &ExternalQMRunner::getStaticBuild() const
{
    return _staticBuild;
}

/**
 * @brief setter for the script path
 *
 * @param scriptPath
 */
void ExternalQMRunner::setScriptPath(const std::string_view &scriptPath)
{
    _scriptPath = scriptPath;
}