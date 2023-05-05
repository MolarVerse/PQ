#ifndef _EXCEPTIONS_H_

#define _EXCEPTIONS_H_

#include <exception>
#include <string>
#include <iostream>

#include "color.hpp"

class CustomException : public std::exception
{
protected:
    std::string _message;

public:
    explicit CustomException(std::string_view message) : _message(message){};
    void colorfulOutput(Color::Code color, std::string_view) const;
};

class InputFileException : public CustomException
{
public:
    explicit InputFileException(const std::string &message) : CustomException(message){};

    const char *what() const throw() override;
};

#endif