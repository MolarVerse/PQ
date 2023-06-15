#include "trajectoryOutput.hpp"

#include <iomanip>

using namespace std;
using namespace simulationBox;

void TrajectoryOutput::writexyz(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfParticles();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << endl;

    _fp << endl;

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[2];

            _fp << endl;
        }
    }
}

void TrajectoryOutput::writeVelocities(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfParticles();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << endl;

    _fp << endl;

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << scientific;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[0];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[1];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[2];

            _fp << endl;
        }
    }
}

void TrajectoryOutput::writeForces(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfParticles();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << endl;

    _fp << endl;

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[2];

            _fp << endl;
        }
    }
}

void TrajectoryOutput::writeCharges(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfParticles();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << endl;

    _fp << endl;

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);

            _fp << setw(15);
            _fp << right;
            _fp << molecule.getPartialCharge(i);

            _fp << endl;
        }
    }
}