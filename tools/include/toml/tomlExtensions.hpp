#ifndef _TOML_EXTENSIONS_HPP_

#define _TOML_EXTENSIONS_HPP_

#include "toml.hpp"
#include <vector>

namespace tomlExtensions
{
    template <typename T>
    std::vector<T> tomlArrayToVector(const toml::array *array)
    {
        std::vector<T> vec;
        array->for_each(
            [&vec](const toml::value<T> &element)
            {
                vec.push_back(element.value_or(T()));
            });
        return vec;
    }
}

#endif // _TOML_EXTENSIONS_HPP_