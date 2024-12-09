#ifndef __POTENTIAL_HANDLE_TYPES_INL__
#define __POTENTIAL_HANDLE_TYPES_INL__

#include <cassert>
#include <type_traits>

#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "lennardJones.hpp"
#include "orthorhombicBox.hpp"
#include "potential.hpp"
#include "triclinicBox.hpp"

namespace potential
{
    template <typename BoxType>
    void inline image(
        const auto* const boxParams,
        auto&             dx,
        auto&             dy,
        auto&             dz,
        auto&             tx,
        auto&             ty,
        auto&             tz
    )
    {
        if constexpr (std::is_same_v<BoxType, simulationBox::OrthorhombicBox>)
            simulationBox::imageOrthoRhombic(boxParams, dx, dy, dz, tx, ty, tz);
        else if constexpr (std::is_same_v<BoxType, simulationBox::TriclinicBox>)
            simulationBox::imageTriclinic(boxParams, dx, dy, dz, tx, ty, tz);
        else
            static_assert(std::false_type::value, "Unsupported BoxType");
    }

    template <typename CoulombType>
    void inline calculateCoulombPotential(
        double&           coulombEnergy,
        auto&             localForce,
        const auto        distance,
        const auto        coulombPreFactor,
        const auto        coulCutOff,
        const auto* const coulParams
    )
    {
        if constexpr (std::is_same_v<CoulombType, CoulombShiftedPotential>)
            coulombEnergy = calculateCoulombShiftedPotential(
                localForce,
                distance,
                coulombPreFactor,
                coulCutOff,
                coulParams
            );
        else if constexpr (std::is_same_v<CoulombType, CoulombWolf>)
            coulombEnergy = calculateCoulombWolfPotential(
                localForce,
                distance,
                coulombPreFactor,
                coulCutOff,
                coulParams
            );
        else
            static_assert(std::false_type::value, "Unsupported Coulomb type");
    }

    template <typename NonCoulombType>
    void calculateNonCoulombEnergy(
        auto&             nonCoulombEnergy,
        auto&             localForce,
        const auto        distance,
        const auto        distanceSquared,
        const auto        rncCutOff,
        const auto* const nonCoulParams
    )
    {
        if constexpr (std::is_same_v<NonCoulombType, LennardJonesFF>)
            nonCoulombEnergy = calculateLennardJones(
                localForce,
                distance,
                distanceSquared,
                rncCutOff,
                nonCoulParams
            );
        else
            static_assert(
                std::false_type::value,
                "Unsupported NonCoulomb type"
            );
    }
}   // namespace potential

#endif   // __POTENTIAL_HANDLE_TYPES_INL__