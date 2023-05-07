#include "postProcessSetup.hpp"
#include "atomMassMap.hpp"
#include "exceptions.hpp"

#include <map>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;

/**
 * @brief Setup post processing
 *
 * @param engine
 */
void postProcessSetup(const Engine &engine)
{
    PostProcessSetup postProcessSetup(engine);
    postProcessSetup.setup();
}

void PostProcessSetup::setup()
{
    setAtomMasses();
}

/**
 * @brief Sets the mass of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void PostProcessSetup::setAtomMasses()
{
    for (Molecule &molecule : _engine._simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));

            if (atomMassMap.find(keyword) == atomMassMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addMass(atomMassMap.at(keyword));
        }
    }
}