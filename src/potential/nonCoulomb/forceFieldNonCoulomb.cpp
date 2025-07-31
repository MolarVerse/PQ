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

#include "forceFieldNonCoulomb.hpp"

#include <algorithm>     // for copy, max
#include <format>        // for std::format
#include <functional>    // for identity
#include <map>           // for map
#include <ranges>        // for __find_if_fn, find_if
#include <string>        // for string
#include <string_view>   // for string_view

#include "exceptions.hpp"       // for ParameterFileException
#include "nonCoulombPair.hpp"   // for NonCoulombPair

using namespace potential;
using namespace customException;
using namespace linearAlgebra;

using std::ranges::adjacent_find;
using std::ranges::find_if;

/**
 * @brief calculates and sets energy and force cutoff for all non-coulombic
 * pairs in _nonCoulPairsVec
 *
 */
void ForceFieldNonCoulomb::setupNonCoulombicCutoffs()
{
    auto setEnergyAndForceCutOff = [](auto &nonCoulombPair)
    {
        const auto radialCutOff     = nonCoulombPair->getRadialCutOff();
        const auto &[energy, force] = nonCoulombPair->calculate(radialCutOff);

        nonCoulombPair->setEnergyCutOff(energy);
        nonCoulombPair->setForceCutOff(force);
    };

    std::ranges::for_each(_nonCoulPairsVec, setEnergyAndForceCutOff);
}

/**
 * @brief determines internal global van der Waals types and sets them in the
 * NonCoulombPair objects
 *
 * @param _externalToInternalGlobalVDWTypes
 *
 */
void ForceFieldNonCoulomb::determineInternalGlobalVdwTypes(
    const std::map<size_t, size_t> &extToIntGlobalVDWTypes
)
{
    auto setIntVDWType = [&extToIntGlobalVDWTypes](auto &nonCoulombPair)
    {
        const auto VDWType1 = nonCoulombPair->getVanDerWaalsType1();
        const auto VDWType2 = nonCoulombPair->getVanDerWaalsType2();

        nonCoulombPair->setInternalType1(extToIntGlobalVDWTypes.at(VDWType1));
        nonCoulombPair->setInternalType2(extToIntGlobalVDWTypes.at(VDWType2));
    };

    std::ranges::for_each(_nonCoulPairsVec, setIntVDWType);
}

/**
 * @brief sorts the elements of a non-coulombic pairs vector
 *
 * @param nonCoulombicPairsVector
 *
 * @throw ParameterFileException if non-coulombic pairs with the same global van
 * der Waals types are defined twice
 */
void ForceFieldNonCoulomb::sortNonCoulombicsPairs(
    std::vector<std::shared_ptr<NonCoulombPair>> &nonCoulPairsVec
)
{
    auto isLess = [](const auto &nonCoulPair1, const auto &nonCoulPair2)
    {
        const auto pair1IntType1 = nonCoulPair1->getInternalType1();
        const auto pair1IntType2 = nonCoulPair1->getInternalType2();
        const auto pair2IntType1 = nonCoulPair2->getInternalType1();
        const auto pair2IntType2 = nonCoulPair2->getInternalType2();

        if (pair1IntType1 < pair2IntType1)
            return true;

        else if (pair1IntType1 == pair2IntType1)
            return pair1IntType2 < pair2IntType2;

        else
            return false;
    };

    std::ranges::sort(nonCoulPairsVec, isLess);

    auto compareSharedPtrs = [](const auto &pair1, const auto &pair2)
    { return *pair1 == *pair2; };

    const auto iter = adjacent_find(nonCoulPairsVec, compareSharedPtrs);

    if (iter != nonCoulPairsVec.end())
        throw ParameterFileException(
            std::format(
                "Non-coulombic pairs with global van der Waals types {} and {} "
                "in "
                "the parameter file are defined twice",
                (*iter)->getVanDerWaalsType1(),
                (*iter)->getVanDerWaalsType2()
            )
        );
}

/**
 * @brief fills the diagonal elements of the non-coulombic pairs matrix
 *
 * @param diagonalElements
 */
void ForceFieldNonCoulomb::fillDiagOfNonCoulPairsMatrix(
    std::vector<std::shared_ptr<NonCoulombPair>> &diag
)
{
    sortNonCoulombicsPairs(diag);

    _nonCoulPairsMat = Matrix<std::shared_ptr<NonCoulombPair>>(diag.size());

    const auto nDiagElements = diag.size();

    for (size_t i = 0; i < nDiagElements; ++i) _nonCoulPairsMat(i, i) = diag[i];
}

/**
 * @brief fills one off-diagonal element of the non-coulombic pairs matrix
 *
 * @details if no type is found check if mixing rules are used
 * if mixing rules are used, then add a new non-coulombic pair with the mixing
 * rule if no mixing rules are used, then throw an
 * exception if a type is found, then add the non-coulombic pair to the matrix
 * if a type is found with both index combinations and it has the same
 * parameters, then add the non-coulombic pair if a type is found with both
 * index combinations and it has different parameters, then throw an exception
 *
 * @param atomType1
 * @param atomType2
 *
 * @throws ParameterFileException if mixing rules are not used
 * and not all combinations of global van der Waals types are defined in the
 * parameter file
 * @throws ParameterFileException if type is found with both
 * index combinations and it has different parameters
 */
void ForceFieldNonCoulomb::setOffDiagonalElement(
    const size_t atomType1,
    const size_t atomType2
)
{
    auto nonCoulPair1 = findNonCoulPairByInternalTypes(atomType1, atomType2);
    auto nonCoulPair2 = findNonCoulPairByInternalTypes(atomType2, atomType1);

    if (nonCoulPair1 == std::nullopt && nonCoulPair2 == std::nullopt)
    {
        if (_mixingRule == MixingRule::NONE)
        {
            throw ParameterFileException(
                "Not all combinations of global van der Waals types are "
                "defined in the parameter file - and no mixing "
                "rules were chosen"
            );
        }
    }
    else if (nonCoulPair1 != std::nullopt && nonCoulPair2 != std::nullopt)
    {
        if (**nonCoulPair1 != **nonCoulPair2)
        {
            const auto vdwType1 = (*nonCoulPair1)->getVanDerWaalsType1();
            const auto vdwType2 = (*nonCoulPair1)->getVanDerWaalsType2();
            throw ParameterFileException(
                std::format(
                    "Non-coulombic pairs with global van der Waals types {}, "
                    "{} "
                    "and {}, {} in the parameter file have "
                    "different parameters",
                    vdwType1,
                    vdwType2,
                    vdwType2,
                    vdwType1
                )
            );
        }

        _nonCoulPairsMat(atomType1, atomType2) = *nonCoulPair1;
        _nonCoulPairsMat(atomType2, atomType1) = *nonCoulPair1;
    }
    else if (nonCoulPair1 != std::nullopt)
    {
        _nonCoulPairsMat(atomType1, atomType2) = *nonCoulPair1;
        _nonCoulPairsMat(atomType2, atomType1) = *nonCoulPair1;
    }
    else
    {
        _nonCoulPairsMat(atomType1, atomType2) = *nonCoulPair2;
        _nonCoulPairsMat(atomType2, atomType1) = *nonCoulPair2;
    }
}

/**
 * @brief fills the off-diagonal elements of the non-coulombic pairs matrix
 *
 */
void ForceFieldNonCoulomb::fillOffDiagOfNonCoulPairsMatrix()
{
    const auto &[rows, cols] = _nonCoulPairsMat.shape();

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = i + 1; j < cols; ++j) setOffDiagonalElement(i, j);
}

/**
 * @brief finds all non-coulombic pairs that are self-interaction pairs
 *
 * @details self interacting means that both internal or external types are the
 * same
 *
 * @return std::vector<std::shared_ptr<NonCoulombPair>>
 */
std::vector<std::shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::
    getSelfInteractionNonCoulPairs() const
{
    auto isSelfInteractionElement = [](const auto &nonCoulombPair)
    {
        const auto internalType1 = nonCoulombPair->getInternalType1();
        const auto internalType2 = nonCoulombPair->getInternalType2();

        return internalType1 == internalType2;
    };

    auto view = _nonCoulPairsVec | std::views::filter(isSelfInteractionElement);

    return std::vector(view.begin(), view.end());
}

/**
 * @brief finds a non coulombic pair by internal types
 *
 * @details if the non coulombic pair is not found, an empty optional is
 * returned if the non coulombic pair is found twice an exception is thrown
 *
 * @param internalType1
 * @param internalType2
 * @return optional<std::shared_ptr<NonCoulombPair>>
 *
 * @throws if the non coulombic pair is found twice
 */
std::optional<std::shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::
    findNonCoulPairByInternalTypes(
        const size_t intType1,
        const size_t intType2
    ) const
{
    auto findByIntAtomTypes = [intType1, intType2](const auto &nonCoulPair)
    {
        const auto internalType1_ = nonCoulPair->getInternalType1();
        const auto internalType2_ = nonCoulPair->getInternalType2();

        auto isEqual = true;
        isEqual      = isEqual && internalType1_ == intType1;
        isEqual      = isEqual && internalType2_ == intType2;

        return isEqual;
    };

    const auto firstNonCoulPair = find_if(_nonCoulPairsVec, findByIntAtomTypes);

    if (firstNonCoulPair != _nonCoulPairsVec.end())
    {
        const auto secondNonCoulPair = find_if(
            firstNonCoulPair + 1,
            _nonCoulPairsVec.end(),
            findByIntAtomTypes
        );

        if (secondNonCoulPair != _nonCoulPairsVec.end())
        {
            auto vdwType1 = (*firstNonCoulPair)->getVanDerWaalsType1();
            auto vdwType2 = (*firstNonCoulPair)->getVanDerWaalsType2();
            throw ParameterFileException(
                std::format(
                    "Non coulombic pair with global van der waals types {} and "
                    "{} "
                    "is defined twice in the parameter file.",
                    vdwType1,
                    vdwType2
                )
            );
        }

        return *firstNonCoulPair;
    }
    else
        return std::nullopt;
}

/**
 * @brief adds a non-coulombic pair to the vector of non-coulombic pairs
 *
 * @param pair
 */
void ForceFieldNonCoulomb::addNonCoulombicPair(
    const std::shared_ptr<NonCoulombPair> &pair
)
{
    _nonCoulPairsVec.push_back(pair);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief gets a shared pointer to a NonCoulombPair object
 *
 * @details the indices vector contains the indices of the atoms in the system,
 * the last two indices are the global van der Waals
 *
 * @param indices
 * @return std::shared_ptr<NonCoulombPair>
 */
std::shared_ptr<NonCoulombPair> ForceFieldNonCoulomb::getNonCoulPair(
    const std::vector<size_t> &indices
)
{
    const auto idx1 = getGlobalVdwType1(indices);
    const auto idx2 = getGlobalVdwType2(indices);

    return _nonCoulPairsMat(idx1, idx2);
}

/**
 * @brief get the global van der Waals type 1
 *
 * @param indices
 * @return size_t
 */
size_t ForceFieldNonCoulomb::getGlobalVdwType1(
    const std::vector<size_t> &indices
) const
{
    return indices[4];
}

/**
 * @brief get the global van der Waals type 2
 *
 * @param indices
 * @return size_t
 */
size_t ForceFieldNonCoulomb::getGlobalVdwType2(
    const std::vector<size_t> &indices
) const
{
    return indices[5];
}

/**
 * @brief Get the Non Coulomb Pairs Vector object
 *
 * @return std::vector<std::shared_ptr<NonCoulombPair>>&
 */
std::vector<std::shared_ptr<NonCoulombPair>> &ForceFieldNonCoulomb::
    getNonCoulombPairsVector()
{
    return _nonCoulPairsVec;
}

/**
 * @brief Get the Non Coulomb Pairs Matrix object
 *
 * @return Matrix<std::shared_ptr<NonCoulombPair>>&
 */
Matrix<std::shared_ptr<NonCoulombPair>> &ForceFieldNonCoulomb::
    getNonCoulombPairsMatrix()
{
    return _nonCoulPairsMat;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the non-coulombic pairs vector
 *
 * @param vec
 */
void ForceFieldNonCoulomb::setNonCoulombPairsVector(
    const std::vector<std::shared_ptr<NonCoulombPair>> &vec
)
{
    _nonCoulPairsVec = vec;
}

/**
 * @brief set the non-coulombic pairs matrix
 *
 * @param mat
 */
void ForceFieldNonCoulomb::setNonCoulombPairsMatrix(
    const Matrix<std::shared_ptr<NonCoulombPair>> &mat
)
{
    _nonCoulPairsMat = mat;
}