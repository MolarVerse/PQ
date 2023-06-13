#include "atom.hpp"

#include <boost/algorithm/string.hpp>

Atom::Atom(const std::string &atomName) : _atomName(atomName)
{
    _elementType = boost::algorithm::to_lower_copy(atomName);
}