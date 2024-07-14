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

#ifndef _Force_FIELD_CLASS_HPP_

#define _Force_FIELD_CLASS_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "angleForceField.hpp"
#include "angleType.hpp"
#include "bondForceField.hpp"
#include "bondType.hpp"
#include "dihedralForceField.hpp"
#include "dihedralType.hpp"
#include "jCouplingForceField.hpp"
#include "jCouplingType.hpp"
#include "typeAliases.hpp"

namespace forceField
{
    /**
     * @class ForceField
     *
     * @brief force field object containing all force field information
     *
     */
    class ForceField
    {
       private:
        bool _isNonCoulombicActivated = false;

        std::vector<BondForceField>      _bonds;
        std::vector<AngleForceField>     _angles;
        std::vector<DihedralForceField>  _dihedrals;
        std::vector<DihedralForceField>  _improperDihedrals;
        std::vector<JCouplingForceField> _jCouplings;

        std::vector<BondType>      _bondTypes;
        std::vector<AngleType>     _angleTypes;
        std::vector<DihedralType>  _dihedralTypes;
        std::vector<DihedralType>  _improperDihedralTypes;
        std::vector<JCouplingType> _jCouplingTypes;

        std::shared_ptr<pq::NonCoulombPot> _nonCoulombPot;
        std::shared_ptr<pq::CoulombPot>    _coulombPotential;

       public:
        std::shared_ptr<ForceField> clone() const;

        void calculateBondedInteractions(const pq::SimBox &, pq::PhysicalData &);
        void calculateExtraInteractions(const pq::SimBox &, pq::PhysicalData &);

        void calculateBondInteractions(const pq::SimBox &, pq::PhysicalData &);
        void calculateAngleInteractions(const pq::SimBox &, pq::PhysicalData &);
        void calculateDihedralInteractions(const pq::SimBox &, pq::PhysicalData &);
        void calculateImproperDihedralInteractions(const pq::SimBox &, pq::PhysicalData &);
        void calculateJCouplingInteractions(const pq::SimBox &, pq::PhysicalData &);

        const BondType      &findBondTypeById(size_t id) const;
        const AngleType     &findAngleTypeById(size_t id) const;
        const DihedralType  &findDihedralTypeById(size_t id) const;
        const DihedralType  &findImproperTypeById(size_t id) const;
        const JCouplingType &findJCouplingTypeById(size_t id) const;

        /*****************************
         * standard activate methods *
         *****************************/

        void activateNonCoulombic();
        void deactivateNonCoulombic();

        [[nodiscard]] bool isNonCoulombicActivated() const;

        /***********************************
         * standard add ForceField Objects *
         ***********************************/

        void addBond(const BondForceField &bond);
        void addAngle(const AngleForceField &angle);
        void addDihedral(const DihedralForceField &dihedral);
        void addImproperDihedral(const DihedralForceField &improperDihedral);
        void addJCoupling(const JCouplingForceField &jCoupling);

        /***************************************
         * standard add ForceFieldType objects *
         ***************************************/

        void addBondType(const BondType &bondType);
        void addAngleType(const AngleType &angleType);
        void addDihedralType(const DihedralType &dihedralType);
        void addImproperDihedralType(const DihedralType &improperType);
        void addJCouplingType(const JCouplingType &jCouplingType);

        /**************************
         * standard clear methods *
         **************************/

        void clearBondTypes();
        void clearAngleTypes();
        void clearDihedralTypes();
        void clearImproperDihedralTypes();
        void clearJCouplingTypes();

        /********************
         * standard setters *
         ********************/

        void setNonCoulombPotential(const pq::SharedNonCoulombPot &pot);
        void setCoulombPotential(const pq::SharedCoulombPot &pot);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] std::vector<BondForceField>      &getBonds();
        [[nodiscard]] std::vector<AngleForceField>     &getAngles();
        [[nodiscard]] std::vector<DihedralForceField>  &getDihedrals();
        [[nodiscard]] std::vector<DihedralForceField>  &getImproperDihedrals();
        [[nodiscard]] std::vector<JCouplingForceField> &getJCouplings();

        [[nodiscard]] const std::vector<BondType>     &getBondTypes() const;
        [[nodiscard]] const std::vector<AngleType>    &getAngleTypes() const;
        [[nodiscard]] const std::vector<DihedralType> &getDihedralTypes() const;
        [[nodiscard]] const std::vector<DihedralType> &getImproperTypes() const;
        [[nodiscard]] const std::vector<JCouplingType> &getJCouplTypes() const;
    };

}   // namespace forceField

#endif   // _Force_FIELD_CLASS_HPP_