#ifndef TOML_H
#define TOML_H

#include <string>

#define TOML_EXCEPTIONS 0
#include "toml++/toml.hpp"

namespace Toml {

    using toml::table;

    table parse_and_bcast(const std::string& filename);
}

#endif
