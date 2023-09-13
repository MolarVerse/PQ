#include "trajOutput.hpp"

#include "frame.hpp"      // for Frame
#include "molecule.hpp"   // for Molecule
#include "vector3d.hpp"   // for Vec3D, operator<<

#include <ostream>   // for operator<<, basic_ostream, char_traits, flush
#include <vector>    // for vector

using namespace frameTools;

void TrajOutput::write(Frame &frame)
{
    const auto &molecules = frame.getMolecules();
    const auto &box       = frame.getBox();

    _fp << molecules.size() << " " << box << '\n';
    _fp << '\n' << std::flush;

    for (const auto &molecule : molecules)
    {
        const auto &com = molecule.getCenterOfMass();

        _fp << "COM " << com[0] << " " << com[1] << " " << com[2] << '\n' << std::flush;
    }
}