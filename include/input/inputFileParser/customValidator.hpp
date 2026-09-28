#ifndef _CUSTOM_VALIDATOR_HPP_
#define _CUSTOM_VALIDATOR_HPP_

#include "keyValidatorBase.hpp"

namespace input
{
    /**
     * Custom validator that uses a user-provided validation function.
     *
     * @tparam T The type of the value to validate.
     */
    template <typename T>
    class CustomValidator : public KeyValidator<T>
    {
       private:
        std::function<bool(const T &)> _validationFunction;
        std::string                    _errorMessage;

       public:
        CustomValidator(
            std::function<bool(const T &)> validationFunction,
            std::string                    errorMessage
        );

        [[nodiscard]]
        bool validate(const T &value) override;

        [[nodiscard]]
        std::string errorMessage() override;
    };
}   // namespace input

#ifndef _CUSTOM_VALIDATOR_TPP_
#include "customValidator.tpp"
#endif

#endif   // _CUSTOM_VALIDATOR_HPP_
