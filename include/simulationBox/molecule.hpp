#ifndef _MOLECULE_H_

#define _MOLECULE_H_

#include <string>
#include <vector>

class Molecule
{
private:
    std::string _name;

    int _moltype;
    int _numberOfAtoms;
    double _charge;

    std::vector<std::string> _atomNames;
    std::vector<int> _atomTypes;
    std::vector<double> _partialCharges;
    // std::vector<double> _positions;
    // std::vector<double> _velocities;
    // std::vector<double> _forces;

    // std::vector<double> centerOfMass;

public:
    explicit Molecule(std::string_view name) : _name(name){};
    explicit Molecule(int moltype) : _moltype(moltype){};

    void setName(std::string_view name) { _name = name; };
    std::string getName() const { return _name; };

    void setMoltype(int moltype) { _moltype = moltype; };
    int getMoltype() const { return _moltype; };

    void setNumberOfAtoms(int numberOfAtoms);
    int getNumberOfAtoms() const { return _numberOfAtoms; };

    void setCharge(double charge) { _charge = charge; };
    double getCharge() const { return _charge; };
};

#endif
