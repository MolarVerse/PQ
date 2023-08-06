#include "configurationReader.hpp"

#include "atom.hpp"

#include <iostream>
#include <sstream>

using namespace std;
using namespace frameTools;
using namespace linearAlgebra;

ConfigurationReader::ConfigurationReader(const vector<string> &filenames) : _filenames(filenames)
{
    _fp.open(_filenames[0], ios::in);
}

bool ConfigurationReader::nextFrame()
{
    if (_fp.peek() == EOF)
    {
        _fp.close();
        _filenames.erase(_filenames.begin());
        if (_filenames.empty())
            return false;
        else
        {
            _fp.open(_filenames[0]);
            return true;
        }
    }

    return true;
}

Frame &ConfigurationReader::getFrame()
{
    _frame = Frame();

    parseHeader();
    parseAtoms();

    _nFrames++;

    return _frame;
}

void ConfigurationReader::parseHeader()
{
    string line;
    getline(_fp, line);

    istringstream iss(line.data());

    size_t nAtoms;
    Vec3D  box;

    iss >> nAtoms;
    iss >> box[0] >> box[1] >> box[2];

    getline(_fp, line);

    // just dummy reader here not implemented yet
    _extxyzReader.readLatticeVectors();

    if (_nFrames != 0)
    {
        if (_nAtoms != nAtoms) throw runtime_error("Number of atoms in the trajectory is not consistent");

        if (isBoxSet(box)) _box = box;
    }
    else
    {
        if (!isBoxSet(box))
            throw runtime_error("Box is not set in first frame");
        else
            _box = box;

        _nAtoms = nAtoms;
    }

    _nFrames++;

    _frame.setBox(_box);
    _frame.setNAtoms(nAtoms);
}

void ConfigurationReader::parseAtoms()
{
    for (size_t i = 0; i < _nAtoms; ++i)
    {
        string line;
        string atomName;
        Vec3D  position;

        getline(_fp, line);

        istringstream iss(line.data());

        iss >> atomName;
        iss >> position[0] >> position[1] >> position[2];

        Atom atom(atomName);
        atom.setPosition(position);

        _frame.addAtom(atom);
    }
}