#include "exceptions.hpp"

void CustomException::colorfulOutput(Color::Code color, std::string_view exception) const
{
    Color::Modifier modifier(color);
    Color::Modifier def(Color::FG_DEFAULT);

    std::cout << modifier << exception << def << std::endl;
}

const char *InputFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "InputFileError");
    return _message.c_str();
}