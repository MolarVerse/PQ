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
#include <cmath>        // for isnan, isinf
#include <filesystem>   // for is_regular_file, path
#include <format>       // for format
#include <fstream>      // for ofstream
#include <string>       // for string
#include <thread>       // for sleep_for

#include "constants/conversionFactors.hpp"   // for _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_, _HARTREE_TO_KCAL_PER_MOL_
#include "exceptions.hpp"                    // for InputFileException
#include "executablePath.hpp"                // for executablePath
#include "fileSettings.hpp"                  // for FileSettings
#include "physicalData.hpp"                  // for PhysicalData
#include "qmSettings.hpp"                    // for QMSettings
#include "simulationBox.hpp"                 // for SimulationBox

using QM::ExternalQMRunner;
using namespace simulationBox;
using namespace physicalData;
using namespace customException;
using namespace settings;
using namespace constants;

std::string QM::bundledQMScriptPath(const std::string_view script)
{
    const auto installedPath = utilities::installedDataPath(
        std::filesystem::path("scripts") / script
    );
    if (std::filesystem::is_regular_file(installedPath))
        return installedPath.string();

    return (std::filesystem::path(SCRIPT_PATH_) / script).string();
}

/**
 * @brief run the qm engine
 *
 * @param simBox
 */
void ExternalQMRunner::run(SimulationBox &simBox, PhysicalData &physicalData)
{
    writeCoordsFile(simBox);

    std::jthread timeoutThread{[this](const std::stop_token stopToken)
                               { throwAfterTimeout(stopToken); }};

    execute();

    timeoutThread.request_stop();

    readForceFile(simBox, physicalData);

    readChargeFile(simBox);

    readStressTensor(simBox.getBox(), physicalData);
}

std::string ExternalQMRunner::resolveScriptPath(
    const std::string_view script
) const
{
    if (_scriptPath.empty())
        return std::string(script);

    if (_scriptPath == SCRIPT_PATH_)
        return bundledQMScriptPath(script);

    return _scriptPath + std::string(script);
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
    const auto forceFileName = FileSettings::getQMForcesTempFileName();

    std::ifstream forceFile(forceFileName);

    if (!forceFile.is_open())
        throw QMRunnerException(
            std::format(
                "Cannot open {} force file \"{}\"",
                string(QMSettings::getQMMethod()),
                forceFileName
            )
        );

    if (forceFile.peek() == std::ifstream::traits_type::eof())
        throw QMRunnerException(
            std::format(
                "Empty {} force file \"{}\"",
                string(QMSettings::getQMMethod()),
                forceFileName
            )
        );

    double energy = 0.0;

    forceFile >> energy;

    if (std::isnan(energy) || std::isinf(energy))
        throw QMRunnerException(std::format(
            "Invalid QM energy (NaN/Inf) in {} force file \"{}\"",
            string(QMSettings::getQMMethod()),
            forceFileName
        ));

    physicalData.setQMEnergy(energy * _HARTREE_TO_KCAL_PER_MOL_);

    auto readForces = [&forceFile, &forceFileName](auto &atom)
    {
        auto grad = linearAlgebra::Vec3D();

        forceFile >> grad[0] >> grad[1] >> grad[2];

        for (size_t i = 0; i < 3; ++i)
            if (std::isnan(grad[i]) || std::isinf(grad[i]))
                throw QMRunnerException(std::format(
                    "Invalid QM force component (NaN/Inf) in {} force file "
                    "\"{}\"",
                    string(QMSettings::getQMMethod()),
                    forceFileName
                ));

        atom->setForce(-grad * _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_);
    };

    std::ranges::for_each(box.getQMAtoms(), readForces);

    forceFile.close();

    if (QMSettings::getRemoveNetForce())
        box.removeNetForce();
}

/**
 * @brief reads the charge file (qm_charges) and sets the _qmCharge of the atoms
 *
 * @param box
 *
 * @throw QMRunnerException
 *  - if the charge file cannot be opened
 *  - if the charge file is empty
 */
void ExternalQMRunner::readChargeFile(SimulationBox &box)
{
    const auto chargeFileName = FileSettings::getQMChargesTempFileName();

    std::ifstream chargeFile(chargeFileName);

    if (!chargeFile.is_open())
        throw QMRunnerException(
            std::format(
                "Cannot open {} charge file \"{}\"",
                string(QMSettings::getQMMethod()),
                chargeFileName
            )
        );

    if (chargeFile.peek() == std::ifstream::traits_type::eof())
        throw QMRunnerException(
            std::format(
                "Empty {} charge file \"{}\"",
                string(QMSettings::getQMMethod()),
                chargeFileName
            )
        );

    box.resetQMCharges();

    auto readCharges = [&chargeFile](auto &atom)
    {
        auto index  = 0;     // Read and discard the first column (index)
        auto charge = 0.0;   // Read the second column (charge value)

        chargeFile >> index >> charge;

        atom->setQMCharge(charge);
    };

    std::ranges::for_each(box.getQMAtoms(), readCharges);

    chargeFile.close();
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
