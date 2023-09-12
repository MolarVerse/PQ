#include "atom.hpp"

#include <boost/algorithm/string/case_conv.hpp>   // for to_lower_copy
#include <boost/iterator/iterator_facade.hpp>     // for operator!=

using namespace frameTools;

Atom::Atom(const std::string &atomName) : _atomName(atomName) { _elementType = boost::algorithm::to_lower_copy(atomName); }