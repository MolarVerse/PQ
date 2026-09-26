#include "inputKeyBase.hpp"

#include "exceptions.hpp"

namespace input
{
    /**
     * @brief constructs a DeprecatedInputKey with the given key and message
     *
     * @param key the key that is deprecated
     * @param message the deprecation message
     */
    DeprecatedInputKey::DeprecatedInputKey(std::string key, std::string message)
        : _key(std::move(key)), _message(std::move(message))
    {
    }

    /**
     * @brief marks the deprecated key as used and prints the deprecation
     * message
     *
     * @throws exc::InputFileException always, indicating that the deprecated
     * key was used
     */
    void DeprecatedInputKey::deprecated(size_t lineNumber) const
    {
        throw exc::InputFileException(
            "Deprecated key '" + _key + "' used at line " +
            std::to_string(lineNumber) + ".\n" + _message
        );
    }

    /**
     * @brief returns the key of the deprecated input key
     *
     * @return the key of the deprecated input key
     */
    const std::string &DeprecatedInputKey::getKey() const { return _key; }
}   // namespace input
