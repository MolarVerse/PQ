add_library(config_headers
    typeAliases.cpp
)

target_link_libraries(config_headers
    PUBLIC
    linearAlgebra
)