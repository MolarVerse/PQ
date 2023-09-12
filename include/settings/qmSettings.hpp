#ifndef _QM_SETTINGS_HPP_

#define _QM_SETTINGS_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string

namespace settings
{
    /**
     * @class enum QMMethod
     *
     */
    enum class QMMethod : size_t
    {
        NONE,
        DFTBPLUS,
    };

    /**
     * @class QMSettings
     *
     * @brief stores all information about the external qm runner
     *
     */
    class QMSettings
    {
      private:
        static inline QMMethod    _qmMethod = QMMethod::NONE;
        static inline std::string _qmScript = "";

      public:
        static void setQMMethod(const std::string_view &method);

        static void setQMMethod(const QMMethod method) { _qmMethod = method; }
        static void setQMScript(const std::string_view &script) { _qmScript = script; }

        [[nodiscard]] static QMMethod getQMMethod() { return _qmMethod; }
        [[nodiscard]] std::string     getQMScript() { return _qmScript; }
    };
}   // namespace settings

#endif   // _QM_SETTINGS_HPP_