#ifndef _EXCEPTIONS_HPP_

#define _EXCEPTIONS_HPP_

#include <exception>
#include <iostream>
#include <string>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "color.hpp"

namespace customException
{
    class CustomException;
    class InputFileException;
    class RstFileException;
    class UserInputException;
    class MolDescriptorException;
    class UserInputExceptionWarning;
    class GuffDatException;
    class TopologyException;
    class ParameterFileException;
    class ManostatException;
    class IntraNonBondedException;
    enum class ExceptionType : size_t;
}   // namespace customException

/**
 * @enum ExceptionType
 *
 */
enum class customException::ExceptionType : size_t
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
class customException::CustomException : public std::exception
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
class customException::InputFileException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class RstFileException inherits from CustomException
 *
 * @brief Exception for restart file errors
 *
 */
class customException::RstFileException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class UserInputException inherits from CustomException
 *
 * @brief Exception for user input errors (CLI)
 *
 */
class customException::UserInputException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class MolDescriptorException inherits from CustomException
 *
 * @brief Exception for MolDescriptor errors
 *
 */
class customException::MolDescriptorException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class UserInputExceptionWarning inherits from CustomException
 *
 * @brief Exception for user input warnings
 *
 */
class customException::UserInputExceptionWarning : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class GuffDatException inherits from CustomException
 *
 * @brief Exception for guff.dat errors
 *
 */
class customException::GuffDatException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class TopologyException inherits from CustomException
 *
 * @brief Exception for topology file errors
 */
class customException::TopologyException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class ParameterFileException inherits from CustomException
 *
 * @brief Exception for parameter file errors
 */
class customException::ParameterFileException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class ManostatException inherits from CustomException
 *
 * @brief Exception for manostat errors
 */
class customException::ManostatException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

/**
 * @class IntraNonBondedException inherits from CustomException
 *
 * @brief Exception for intra non bonded errors
 */
class customException::IntraNonBondedException : public customException::CustomException
{
  public:
    using customException::CustomException::CustomException;

    const char *what() const throw() override;
};

#endif   // _EXCEPTIONS_HPP_