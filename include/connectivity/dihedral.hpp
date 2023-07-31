#ifndef _DIHEDRAL_HPP_

#define _DIHEDRAL_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    class Dihedral;
}

/**
 * @class Dihedral
 *
 * @brief dihedral object containing all dihedral information
 *
 */
class connectivity::Dihedral : public connectivity::ConnectivityElement
{
  public:
    using connectivity::ConnectivityElement::ConnectivityElement;
};

#endif   // _DIHEDRAL_HPP_