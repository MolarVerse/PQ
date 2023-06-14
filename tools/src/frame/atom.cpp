#include "atom.hpp"

#include <boost/algorithm/string.hpp>

using namespace frameTools;

Atom::Atom(const std::string &atomName) : _atomName(atomName)
{
    _elementType = boost::algorithm::to_lower_copy(atomName);
}