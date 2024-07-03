#include "velocityVerlet.hpp"

#include "simulationBox.hpp"

using namespace integrator;
using namespace simulationBox;

VelocityVerlet::VelocityVerlet() : Integrator("VelocityVerlet"){};

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(SimulationBox &simBox)
{
    startTimingsSection("Velocity Verlet - First Step");

    auto integrate = [this, &simBox](auto &atom)
    {
        integrateVelocities(atom.get());
        integratePositions(atom.get(), simBox);
    };

    std::ranges::for_each(simBox.getAtoms(), integrate);

    const auto box = simBox.getBoxPtr();

    auto calculateCOM = [&box](auto &molecule)
    {
        molecule.calculateCenterOfMass(*box);
        molecule.setAtomForcesToZero();
    };

    std::ranges::for_each(simBox.getMolecules(), calculateCOM);

    stopTimingsSection("Velocity Verlet - First Step");
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox &simBox)
{
    startTimingsSection("Velocity Verlet - Second Step");

    std::ranges::for_each(
        simBox.getAtoms(),
        [this](auto atom) { integrateVelocities(atom.get()); }
    );

    stopTimingsSection("Velocity Verlet - Second Step");
}