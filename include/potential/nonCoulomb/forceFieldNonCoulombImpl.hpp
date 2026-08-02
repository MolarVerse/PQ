#ifndef _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_
#define _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_

#include "forceFieldNonCoulomb.hpp"
#include "matrix.hpp"

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

/**
 * @brief struct to hold the non-coulombic pairs matrix
 *
 */
struct potential::ForceFieldNonCoulomb::matrix
{
    linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> matrix;
};

#endif   // _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_
