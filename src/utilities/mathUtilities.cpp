#include "mathUtilities.hpp"

/**
 * @brief specializing of template function compare
 *
 * @param a
 * @param b
 * @return true
 * @return false
 */
bool utilities::compare(const linearAlgebra::Vec3D &a, const linearAlgebra::Vec3D &b)
{
    return compare<double>(a[0], b[0]) && compare<double>(a[1], b[1]) && compare<double>(a[2], b[2]);
}