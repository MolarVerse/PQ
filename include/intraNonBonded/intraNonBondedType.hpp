#ifndef _INTRA_NON_BONDED_TYPE_HPP_

#define _INTRA_NON_BONDED_TYPE_HPP_

#include <cstddef>

namespace intraNonBonded
{
    class IntraNonBondedType;
}   // namespace intraNonBonded

/**
 * @class IntraNonBondedType
 *
 * @brief represents a container for a single intra non bonded type
 */
class intraNonBonded::IntraNonBondedType
{
  private:
    size_t _molType;
    size_t _atomIndex1;
    size_t _atomIndex2;

  public:
    IntraNonBondedType(const size_t molType, const size_t atomIndex1, const size_t atomIndex2)
        : _molType(molType), _atomIndex1(atomIndex1), _atomIndex2(atomIndex2){};

    [[nodiscard]] size_t getMolType() const { return _molType; }
    [[nodiscard]] size_t getAtomIndex1() const { return _atomIndex1; }
    [[nodiscard]] size_t getAtomIndex2() const { return _atomIndex2; }
};

#endif   // _INTRA_NON_BONDED_TYPE_HPP_
