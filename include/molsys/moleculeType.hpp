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

#ifndef _MOLECULE_TYPE_HPP_

#define _MOLECULE_TYPE_HPP_

#include <map>           // for map
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "strongTypes.hpp"

namespace molsys
{
    /**
     * @class MoleculeType
     *
     * @brief containing all information about a molecule type
     */
    class MoleculeType
    {
       private:
        std::string _name;

        MolType _moltype;
        size_t  _numberOfAtoms;

        int _charge;

        std::vector<std::string> _atomNames;
        std::vector<AtomType>    _atomTypes;
        std::vector<ExtAtomType> _externalAtomTypes;
        std::vector<ExtVdwType>  _externalGlobalVDWTypes;
        std::vector<double>      _partialCharges;

        std::map<ExtAtomType, AtomType> _externalToInternalAtomTypes;

       public:
        MoleculeType() = default;
        explicit MoleculeType(MolType moltype);
        explicit MoleculeType(const std::string_view &name);

        [[nodiscard]] size_t getNumberOfAtomTypes();

        /**************************
         * standard adder methods *
         **************************/

        void addAtomName(const std::string &atomName);
        void addExternalAtomType(const ExtAtomType externalAtomType);
        void addPartialCharge(const double partialCharge);
        void addExternalGlobalVDWType(const ExtVdwType externalGlobalVDWType);

        void addExternalToInternalAtomTypeElement(ExtAtomType, AtomType);
        void addAtomType(AtomType atomType);

        /***************************
         * standard setter methods *
         ***************************/

        void setName(const std::string_view &name);

        void setNumberOfAtoms(const size_t numberOfAtoms);
        void setMoltype(MolType moltype);

        void setCharge(const int charge);
        void setPartialCharge(AtomIndex index, const double partialCharge);
        void setPartialCharges(const std::vector<double> &partialCharges);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t      getNumberOfAtoms() const;
        [[nodiscard]] MolType     getMoltype() const;
        [[nodiscard]] ExtAtomType getExternalAtomType(AtomIndex index) const;
        [[nodiscard]] AtomType    getAtomType(AtomIndex index) const;
        [[nodiscard]] AtomType    getInternalAtomType(
               const ExtAtomType type
           ) const;

        [[nodiscard]] int    getCharge() const;
        [[nodiscard]] double getPartialCharge(AtomIndex index) const;

        [[nodiscard]] std::string getName() const;
        [[nodiscard]] std::string getAtomName(AtomIndex index) const;

        [[nodiscard]] std::vector<std::string>  getAtomNames() const;
        [[nodiscard]] std::vector<ExtAtomType> &getExternalAtomTypes();
        [[nodiscard]] std::vector<ExtVdwType>  &getExternalGlobalVDWTypes();
        [[nodiscard]] std::vector<double>      &getPartialCharges();

        [[nodiscard]]
        const std::map<ExtAtomType, AtomType> &getExternalToInternalAtomTypes(
        ) const;
    };

}   // namespace molsys

#endif   // _MOLECULE_TYPE_HPP_
