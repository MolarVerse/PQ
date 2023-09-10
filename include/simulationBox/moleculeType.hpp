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

        double _charge;

        std::vector<std::string> _atomNames;
        std::vector<size_t>      _atomTypes;
        std::vector<size_t>      _externalAtomTypes;
        std::vector<size_t>      _externalGlobalVDWTypes;
        std::vector<double>      _partialCharges;

        std::map<size_t, size_t> _externalToInternalAtomTypes;

      public:
        MoleculeType() = default;
        explicit MoleculeType(const size_t moltype) : _moltype(moltype){};
        explicit MoleculeType(const std::string_view &name) : _name(name){};

        size_t getNumberOfAtomTypes();

        /**************************
         * standard adder methods *
         **************************/

        void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); }
        void addExternalAtomType(const size_t externalAtomType) { _externalAtomTypes.push_back(externalAtomType); }
        void addPartialCharge(const double partialCharge) { _partialCharges.push_back(partialCharge); }
        void addExternalGlobalVDWType(const size_t externalGlobalVDWType)
        {
            _externalGlobalVDWTypes.push_back(externalGlobalVDWType);
        }

        void addExternalToInternalAtomTypeElement(const size_t key, const size_t value)
        {
            _externalToInternalAtomTypes.try_emplace(key, value);
        }
        void addAtomType(const size_t atomType) { _atomTypes.push_back(atomType); }

        /***************************
         * standard setter methods *
         ***************************/

        void setNumberOfAtoms(const size_t numberOfAtoms) { _numberOfAtoms = numberOfAtoms; }
        void setMoltype(const size_t moltype) { _moltype = moltype; }

        void setCharge(const double charge) { _charge = charge; }
        void setPartialCharge(const size_t index, const double partialCharge) { _partialCharges[index] = partialCharge; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNumberOfAtoms() const { return _numberOfAtoms; }
        [[nodiscard]] size_t getMoltype() const { return _moltype; }

        [[nodiscard]] double getCharge() const { return _charge; }
        [[nodiscard]] double getPartialCharge(const size_t index) const { return _partialCharges[index]; }

        [[nodiscard]] std::string getName() const { return _name; }

        [[nodiscard]] size_t getExternalAtomType(const size_t index) const { return _externalAtomTypes[index]; }
        [[nodiscard]] size_t getAtomType(const size_t index) const { return _atomTypes[index]; }

        [[nodiscard]] std::vector<std::string> getAtomNames() const { return _atomNames; }
        [[nodiscard]] std::vector<size_t>      getExternalAtomTypes() const { return _externalAtomTypes; }
        [[nodiscard]] std::vector<double>      getPartialCharges() const { return _partialCharges; }
        [[nodiscard]] std::vector<size_t>      getExternalGlobalVDWTypes() const { return _externalGlobalVDWTypes; }

        [[nodiscard]] size_t getInternalAtomType(const size_t type) const { return _externalToInternalAtomTypes.at(type); }

        [[nodiscard]] std::map<size_t, size_t> getExternalToInternalAtomTypes() const { return _externalToInternalAtomTypes; }
    };

}   // namespace simulationBox

#endif   // _MOLECULE_TYPE_HPP_