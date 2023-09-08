#ifndef _INTRA_NON_BONDED_CONTAINER_HPP_

#define _INTRA_NON_BONDED_CONTAINER_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace intraNonBonded
{
    /**
     * @class IntraNonBondedContainer
     *
     * @brief represents a container for a single intra non bonded type
     */
    class IntraNonBondedContainer
    {
      private:
        size_t                        _molType;
        std::vector<std::vector<int>> _atomIndices;

      public:
        IntraNonBondedContainer(const size_t molType, const std::vector<std::vector<int>> &atomIndices)
            : _molType(molType), _atomIndices(atomIndices){};

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t                        getMolType() const { return _molType; }
        [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const { return _atomIndices; }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_CONTAINER_HPP_
