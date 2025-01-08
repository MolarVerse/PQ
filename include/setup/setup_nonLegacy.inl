#ifndef __SETUP_NON_LEGACY_INL__
#define __SETUP_NON_LEGACY_INL__

#include "buckingham.hpp"
#include "buckinghamPair.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "guffNonCoulomb.hpp"
#include "lennardJones.hpp"
#include "lennardJonesPair.hpp"
#include "morse.hpp"
#include "morsePair.hpp"
#include "potential.hpp"
#include "setup.hpp"

/**
 * @brief setup the flattened ForceFieldNonCoulomb potential
 *
 * @tparam T the type of the potential
 * @param potential the potential to setup
 */
template <typename T>
void setup::setupFlattenedNonCoulPotFF(potential::Potential *const pot)
{
    auto &nonCoulPot = dynamic_cast<potential::ForceFieldNonCoulomb &>(
        pot->getNonCoulombPotential()
    );

    auto &nonCoulPairs    = nonCoulPot.getNonCoulombPairsMatrix();
    const auto [row, col] = nonCoulPairs.shape();

    assert(row == col);

    auto tempPot = T();
    tempPot.resize(row);

    // clang-format off
    // #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < row; ++i)
        for (size_t j = 0; j < col; ++j)
        {
            if constexpr (std::is_same_v<T, potential::LennardJones>)
            {
                const auto &pair =
                    dynamic_cast<potential::LennardJonesPair &>(
                        *nonCoulPairs(i, j)
                    );
                tempPot.addPair(pair, i, j);
            }
            else if constexpr (std::is_same_v<T, potential::Buckingham>)
            {
                const auto &pair =
                    dynamic_cast<potential::BuckinghamPair &>(
                        *nonCoulPairs(i, j)
                    );
                tempPot.addPair(pair, i, j);
            }
            else if constexpr (std::is_same_v<T, potential::Morse>)
            {
                const auto &pair = dynamic_cast<potential::MorsePair &>(
                    *nonCoulPairs(i, j)
                );
                tempPot.addPair(pair, i, j);
            }
            else
                static_assert(std::is_same_v<T, void>,
                                "Unknown potential type");
        }
    // clang-format on

    const auto params  = tempPot.copyParams();
    const auto cutOffs = tempPot.copyCutOffs();
    const auto size    = tempPot.getSize();
    const auto nParams = T::getNParams();

    pot->setNonCoulombParamVectors(params, cutOffs, nParams, size);
}

/**
 * @brief setup the flattened guff non-Coulomb potential
 *
 * @param engine
 */
template <typename T>
void setup::setupFlattenedNonCoulPotGuff(
    potential::Potential *const   pot,
    simulationBox::SimulationBox &simBox
)
{
    const size_t numMolTypes = simBox.getMoleculeTypes().size();

    size_t maxNumAtomTypes = 0;

    for (size_t i = 0; i < numMolTypes; ++i)
    {
        const size_t numAtomTypes =
            simBox.getMoleculeType(i).getNumberOfAtomTypes();

        if (numAtomTypes > maxNumAtomTypes)
            maxNumAtomTypes = numAtomTypes;
    }

    const auto &nonCoulPot =
        dynamic_cast<potential::GuffNonCoulomb &>(pot->getNonCoulombPotential()
        );

    auto nonCoulPairs = nonCoulPot.getNonCoulombPairs();

    auto tempPot = T();
    tempPot.resize(numMolTypes * maxNumAtomTypes);

    for (size_t i = 0; i < numMolTypes; ++i)
    {
        for (size_t j = 0; j < numMolTypes; ++j)
        {
            auto      &molType1    = simBox.getMoleculeType(i);
            const auto nAtomTypes1 = molType1.getNumberOfAtomTypes();

            for (size_t k = 0; k < nAtomTypes1; ++k)
            {
                auto      &molType2    = simBox.getMoleculeType(j);
                const auto nAtomTypes2 = molType2.getNumberOfAtomTypes();

                for (size_t l = 0; l < nAtomTypes2; ++l)
                {
                    const size_t index_i =
                        i * numMolTypes * numMolTypes * maxNumAtomTypes +
                        j * numMolTypes * maxNumAtomTypes +
                        k * maxNumAtomTypes + l;

                    const size_t index_j =
                        i * numMolTypes * numMolTypes * maxNumAtomTypes +
                        j * numMolTypes * maxNumAtomTypes +
                        l * maxNumAtomTypes + k;

                    const size_t index_k =
                        j * numMolTypes * numMolTypes * maxNumAtomTypes +
                        i * numMolTypes * maxNumAtomTypes +
                        k * maxNumAtomTypes + l;

                    const size_t index_l =
                        j * numMolTypes * numMolTypes * maxNumAtomTypes +
                        i * numMolTypes * maxNumAtomTypes +
                        l * maxNumAtomTypes + k;

                    try
                    {
                        if constexpr (std::
                                          is_same_v<T, potential::LennardJones>)
                        {
                            const auto &pair =
                                dynamic_cast<potential::LennardJonesPair &>(
                                    *nonCoulPairs[i][j][k][l]
                                );

                            tempPot.addPair(pair, index_i, index_j);
                            tempPot.addPair(pair, index_k, index_l);
                        }
                        else if constexpr (std::is_same_v<
                                               T,
                                               potential::Buckingham>)
                        {
                            std::cout << i << j << k << l << std::endl
                                      << std::flush;
                            const auto &pair =
                                dynamic_cast<potential::BuckinghamPair &>(
                                    *nonCoulPairs[i][j][k][l]
                                );

                            tempPot.addPair(pair, index_i, index_j);
                            tempPot.addPair(pair, index_k, index_l);
                        }
                        else if constexpr (std::is_same_v<T, potential::Morse>)
                        {
                            const auto &pair =
                                dynamic_cast<potential::MorsePair &>(
                                    *nonCoulPairs[i][j][k][l]
                                );

                            tempPot.addPair(pair, index_i, index_j);
                            tempPot.addPair(pair, index_k, index_l);
                            // }else if constexpr (std::is_same_v<T,
                            // potential::Guff>)
                            // {
                            //     const auto &pair =
                            //         dynamic_cast<potential::GuffPair &>(
                            //             *nonCoulPairs[i][j][k][l]
                            //         );

                            //     tempPot.addPair(pair, index_i, index_j);
                            //     tempPot.addPair(pair, index_k, index_l);
                        }
                        else
                            static_assert(
                                std::is_same_v<T, void>,
                                "Unknown potential type"
                            );
                    }
                    catch (const std::bad_cast &e)
                    {
                        continue;
                    }
                }
            }
        }
    }

    const auto params  = tempPot.copyParams();
    const auto cutOffs = tempPot.copyCutOffs();
    const auto size    = tempPot.getSize();
    const auto nParams = T::getNParams();

    pot->setNonCoulombParamVectors(params, cutOffs, nParams, size);
    pot->setMaxNumAtomTypes(maxNumAtomTypes);
    pot->setNumMolTypes(numMolTypes);
}

#endif   // __SETUP_NON_LEGACY_INL__
