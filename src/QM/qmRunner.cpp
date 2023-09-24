#include "qmRunner.hpp"

#include "atom.hpp"              // for Atom
#include "constants.hpp"         // for constants
#include "exceptions.hpp"        // for InputFileException
#include "physicalData.hpp"      // for PhysicalData
#include "qmSettings.hpp"        // for QMSettings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for fileExists
#include "vector3d.hpp"          // for Vec3D

#include <algorithm>    // for __for_each_fn, for_each
#include <cstddef>      // for size_t
#include <cstdlib>      // for system
#include <format>       // for format
#include <fstream>      // for ofstream
#include <functional>   // for identity
#include <ranges>       // for borrowed_iterator_t, __distance_fn
#include <string>       // for string
#include <vector>       // for vector

using QM::QMRunner;

/**
 * @brief run the qm engine
 *
 * @param box
 */
void QMRunner::run(simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    writeCoordsFile(box);
    execute();
    readForceFile(box, physicalData);
}

/**
 * @brief reads the force file (including qm energy) and sets the forces of the atoms
 *
 * @param box
 * @param physicalData
 */
void QMRunner::readForceFile(simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    const auto forceFileName = "qm_forces";

    std::ifstream forceFile(forceFileName);

    if (!forceFile.is_open())
        throw customException::QMRunnerException(
            std::format("Cannot open {} force file \"{}\"", string(settings::QMSettings::getQMMethod()), forceFileName));

    double energy = 0.0;

    forceFile >> energy;

    physicalData.setQMEnergy(energy * constants::_HARTREE_TO_KCAL_PER_MOL_);

    auto readForces = [&forceFile](auto &atom)
    {
        auto grad = linearAlgebra::Vec3D();

        forceFile >> grad[0] >> grad[1] >> grad[2];

        atom->setForce(-grad * constants::_HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_);
    };

    std::ranges::for_each(box.getQMAtoms(), readForces);

    forceFile.close();
}