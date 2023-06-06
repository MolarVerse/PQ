#ifndef _FRAME_HPP_

#define _FRAME_HPP_

#include <vector>
#include <string>

#include "vector3d.hpp"

class Frame
{
private:
    size_t _nAtoms;
    Vec3D _box;

    std::vector<std::string> _atomNames;
    std::vector<double> _masses;

    std::vector<Vec3D> _positions;
    std::vector<Vec3D> _velocities;
    std::vector<Vec3D> _forces;

public:
    // standard getter and setter
    void setNAtoms(const size_t nAtoms) { _nAtoms = nAtoms; }

    void setBox(const Vec3D &box) { _box = box; }

    void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); }

    void addPosition(const Vec3D &position) { _positions.push_back(position); }
};

#endif // _FRAME_HPP_