#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include "atom.hpp"
#include "vector3d.hpp"

#include <algorithm>
#include <cstddef>   // for size_t
#include <map>
#include <string>
#include <string_view>   // for string_view
#include <vector>

namespace simulationBox
{
    using c_ul = const size_t;

    /**
     * @class Molecule
     *
     * @brief containing all information about a molecule
     */
    class Molecule
    {
      private:
        std::string _name;
        size_t      _moltype;

        double _charge;   // set via molDescriptor not sum of partial charges!!!
        double _molMass;

        std::map<size_t, size_t> _externalToInternalAtomTypes;

        std::vector<Atom *> _atoms;

        linearAlgebra::Vec3D _centerOfMass = linearAlgebra::Vec3D(0.0, 0.0, 0.0);

      public:
        Molecule() = default;
        explicit Molecule(const std::string_view name) : _name(name){};
        explicit Molecule(c_ul moltype) : _moltype(moltype){};

        void calculateCenterOfMass(const linearAlgebra::Vec3D &);
        void scale(const linearAlgebra::Vec3D &);
        void scaleVelocities(const double scaleFactor);
        void correctVelocities(const linearAlgebra::Vec3D &correction);

        [[nodiscard]] size_t getNumberOfAtomTypes();

        // TODO: check if these are really necessary

        [[nodiscard]] linearAlgebra::Vec3D getAtomPosition(const size_t index) { return _atoms[index]->getPosition(); }

        void setPartialCharges(const std::vector<double> &partialCharges)
        {
            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
            {
                _atoms[i]->setPartialCharge(partialCharges[i]);
            }
        }

        [[nodiscard]] std::vector<double> getPartialCharges() const
        {
            std::vector<double> partialCharges(getNumberOfAtoms());

            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
            {
                partialCharges[i] = _atoms[i]->getPartialCharge();
            }

            return partialCharges;
        }

        [[nodiscard]] std::vector<size_t> getExternalGlobalVDWTypes()
        {
            std::vector<size_t> externalGlobalVDWTypes(getNumberOfAtoms());

            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
            {
                externalGlobalVDWTypes[i] = _atoms[i]->getExternalGlobalVDWType();
            }

            return externalGlobalVDWTypes;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getMoltype() const { return _moltype; }
        [[nodiscard]] size_t getNumberOfAtoms() const { return _atoms.size(); }
        [[nodiscard]] size_t getDegreesOfFreedom() const { return 3 * getNumberOfAtoms(); }

        [[nodiscard]] double getCharge() const { return _charge; }
        [[nodiscard]] double getMolMass() const { return _molMass; }

        [[nodiscard]] std::string getName() const { return _name; }

        [[nodiscard]] Atom *getAtom(const size_t index) { return _atoms[index]; }

        [[nodiscard]] linearAlgebra::Vec3D getCenterOfMass() const { return _centerOfMass; }

        /***************************
         * standard setter methods *
         ***************************/

        void setName(const std::string_view name) { _name = name; }
        void setMoltype(c_ul moltype) { _moltype = moltype; }
        void setCharge(const double charge) { _charge = charge; }
        void setMolMass(const double molMass) { _molMass = molMass; }
        void setCenterOfMass(const linearAlgebra::Vec3D &centerOfMass) { _centerOfMass = centerOfMass; }
    };

}   // namespace simulationBox

#endif   // _MOLECULE_HPP_
