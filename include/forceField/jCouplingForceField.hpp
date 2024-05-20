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

#ifndef _J_COUPLING_FORCE_FIELD_HPP_

#define _J_COUPLING_FORCE_FIELD_HPP_

#include <cstddef>
#include <vector>

#include "dihedral.hpp"

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace forceField
{
    /**
     * @class DihedralForceField
     *
     * @brief Represents a dihedral between four atoms.
     *
     */
    class JCouplingForceField : public connectivity::Dihedral
    {
       private:
        size_t _type;
        bool   _upperSymmetry = true;
        bool   _lowerSymmetry = true;

        double _J0;
        double _forceConstant;
        double _a;
        double _b;
        double _c;
        double _phaseShift;

       public:
        JCouplingForceField(
            const std::vector<simulationBox::Molecule *> &molecules,
            const std::vector<size_t>                    &atomIndices,
            size_t                                        type
        )
            : connectivity::Dihedral(molecules, atomIndices), _type(type){};

        // void calculateEnergyAndForces(const simulationBox::SimulationBox &,
        //                               physicalData::PhysicalData &,
        //                               const bool isImproperDihedral,
        //                               const potential::CoulombPotential &,
        //                               potential::NonCoulombPotential &);

        /***************************
         * standard setter methods *
         ***************************/

        void setUpperSymmetry(const bool boolean) { _upperSymmetry = boolean; }
        void setLowerSymmetry(const bool boolean) { _lowerSymmetry = boolean; }

        void setJ0(const double J0) { _J0 = J0; }
        void setForceConstant(const double k) { _forceConstant = k; }
        void setA(const double a) { _a = a; }
        void setB(const double b) { _b = b; }
        void setC(const double c) { _c = c; }
        void setPhaseShift(const double phi) { _phaseShift = phi; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getType() const { return _type; }

        [[nodiscard]] bool getUpperSymmetry() const { return _upperSymmetry; }
        [[nodiscard]] bool getLowerSymmetry() const { return _lowerSymmetry; }

        [[nodiscard]] double getJ0() const { return _J0; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
        [[nodiscard]] double getA() const { return _a; }
        [[nodiscard]] double getB() const { return _b; }
        [[nodiscard]] double getC() const { return _c; }
        [[nodiscard]] double getPhaseShift() const { return _phaseShift; }
    };

}   // namespace forceField

#endif   // _J_COUPLING_FORCE_FIELD_HPP_
