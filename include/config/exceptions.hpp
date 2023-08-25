#ifndef _EXCEPTIONS_HPP_

#define _EXCEPTIONS_HPP_

#include <cstddef>
#include <exception>
#include <string>
#include <string_view>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "color.hpp"

namespace customException
{

    /**
     * @enum ExceptionType
     *
     */
    enum class ExceptionType : size_t
    {
        INPUTFILEEXCEPTION,
        RSTFILEEXCEPTION,
        USERINPUTEXCEPTION,
        MOLDESCRIPTOREXCEPTION,
        USERINPUTEXCEPTIONWARNING,
        GUFFDATEXCEPTION,
        TOPOLOGYEXCEPTION,
        PARAMETERFILEEXCEPTION,
        MANOSTATEXCEPTION
    };

    /**
     * @class CustomException
     *
     * @brief Custom exception base class
     *
     */
    class CustomException : public std::exception
    {
      protected:
        std::string _message;

      public:
        explicit CustomException(const std::string_view message) : _message(message){};
        void colorfulOutput(const Color::Code color, const std::string_view) const;
    };

    /**
     * @class InputFileException inherits from CustomException
     *
     * @brief Exception for input file errors
     *
     */
    class InputFileException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class RstFileException inherits from CustomException
     *
     * @brief Exception for restart file errors
     *
     */
    class RstFileException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class UserInputException inherits from CustomException
     *
     * @brief Exception for user input errors (CLI)
     *
     */
    class UserInputException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class MolDescriptorException inherits from CustomException
     *
     * @brief Exception for MolDescriptor errors
     *
     */
    class MolDescriptorException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class UserInputExceptionWarning inherits from CustomException
     *
     * @brief Exception for user input warnings
     *
     */
    class UserInputExceptionWarning : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class GuffDatException inherits from CustomException
     *
     * @brief Exception for guff.dat errors
     *
     */
    class GuffDatException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class TopologyException inherits from CustomException
     *
     * @brief Exception for topology file errors
     */
    class TopologyException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ParameterFileException inherits from CustomException
     *
     * @brief Exception for parameter file errors
     */
    class ParameterFileException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ManostatException inherits from CustomException
     *
     * @brief Exception for manostat errors
     */
    class ManostatException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class IntraNonBondedException inherits from CustomException
     *
     * @brief Exception for intra non bonded errors
     */
    class IntraNonBondedException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ShakeException inherits from CustomException
     *
     * @brief Exception for SHAKE errors
     */
    class ShakeException : public CustomException
    {
      public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

}   // namespace customException

#endif   // _EXCEPTIONS_HPP_