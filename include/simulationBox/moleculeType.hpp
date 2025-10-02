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

namespace simulationBox
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

        size_t _moltype;
        size_t _numberOfAtoms;

        int _charge;

        std::vector<std::string> _atomNames;
        std::vector<size_t>      _atomTypes;
        std::vector<size_t>      _externalAtomTypes;
        std::vector<size_t>      _externalGlobalVDWTypes;
        std::vector<double>      _partialCharges;

        std::map<size_t, size_t> _externalToInternalAtomTypes;

       public:
        MoleculeType() = default;
        explicit MoleculeType(const size_t moltype);
        explicit MoleculeType(const std::string_view &name);

        [[nodiscard]] size_t getNumberOfAtomTypes();

        /**************************
         * standard adder methods *
         **************************/

        void addAtomName(const std::string &atomName);
        void addExternalAtomType(const size_t externalAtomType);
        void addPartialCharge(const double partialCharge);
        void addExternalGlobalVDWType(const size_t externalGlobalVDWType);

        void addExternalToInternalAtomTypeElement(const size_t, const size_t);
        void addAtomType(const size_t atomType);

        /***************************
         * standard setter methods *
         ***************************/

        void setName(const std::string_view &name);

        void setNumberOfAtoms(const size_t numberOfAtoms);
        void setMoltype(const size_t moltype);

        void setCharge(const int charge);
        void setPartialCharge(const size_t index, const double partialCharge);
        void setPartialCharges(const std::vector<double> &partialCharges);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNumberOfAtoms() const;
        [[nodiscard]] size_t getMoltype() const;
        [[nodiscard]] size_t getExternalAtomType(const size_t index) const;
        [[nodiscard]] size_t getAtomType(const size_t index) const;
        [[nodiscard]] size_t getInternalAtomType(const size_t type) const;

        [[nodiscard]] int    getCharge() const;
        [[nodiscard]] double getPartialCharge(const size_t index) const;

        [[nodiscard]] std::string getName() const;
        [[nodiscard]] std::string getAtomName(const size_t index) const;

        [[nodiscard]] std::vector<std::string> &getAtomNames();
        [[nodiscard]] std::vector<size_t>      &getExternalAtomTypes();
        [[nodiscard]] std::vector<size_t>      &getExternalGlobalVDWTypes();
        [[nodiscard]] std::vector<double>      &getPartialCharges();

        [[nodiscard]] std::map<size_t, size_t> getExternalToInternalAtomTypes(
        ) const;
    };

}   // namespace simulationBox

#endif   // _MOLECULE_TYPE_HPP_