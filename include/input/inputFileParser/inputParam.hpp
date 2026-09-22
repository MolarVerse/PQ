/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _INPUT_PARAM_HPP_
#define _INPUT_PARAM_HPP_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "inputKeyBase.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "keyValidatorBase.hpp"

namespace input
{

    /**
     * @class InputKey
     *
     * @brief owns key identity, default/actual value storage, and the
     * allowed-values restriction for a single input-file key
     *
     * @details delegates raw-token -> T conversion entirely to
     * Converter<T> (or a per-key customParser override) -- no parsing
     * logic lives here
     *
     * @tparam T
     */
    template <typename T>
    class InputKey : public InputKeyBase
    {
       public:
       private:
        KeyMetadata                           _metadata;
        std::optional<T>                      _default;
        std::optional<T>                      _value;
        std::optional<std::vector<T>>         _allowed;
        typename KeyRegistry<T>::CustomParser _customParser;
        std::function<void(const T &)>        _onSet;
        std::shared_ptr<KeyValidator<T>>      _validator;

       public:
        explicit InputKey(const KeyRegistry<T> &registry);

        void parse(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        ) override;

        [[nodiscard]] const std::string      &name() const override;
        [[nodiscard]] bool                    isSet() const override;
        [[nodiscard]] const std::optional<T> &explicitValue() const;
        [[nodiscard]] const std::optional<T> &defaultValue() const;
        [[nodiscard]] const T                &value() const;
        [[nodiscard]] std::optional<T>        tryValue() const;
        [[nodiscard]] std::string             describe() const override;
        void                                  clearValue() override;

       private:
        [[nodiscard]]
        static std::string _valueToString(const T &value);
    };

}   // namespace input

#ifndef _INPUT_PARAM_TPP_
#include "inputParam.tpp"
#endif

#endif   // _INPUT_PARAM_HPP_
