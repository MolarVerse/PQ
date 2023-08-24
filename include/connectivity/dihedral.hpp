#ifndef _DIHEDRAL_HPP_

#define _DIHEDRAL_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    /**
     * @class Dihedral
     *
     * @brief dihedral object containing all dihedral information
     *
     */
    class Dihedral : public ConnectivityElement
    {
      public:
        using ConnectivityElement::ConnectivityElement;
    };

}   // namespace connectivity

#endif   // _DIHEDRAL_HPP_