# Turns a file into a C++ byte array: cmake -DINPUT=... -DOUTPUT=... -DSYMBOL=... -P embed.cmake
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR size "${hex_len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
# Break the array into lines so compilers and editors stay happy.
string(REGEX REPLACE "((0x..,){32})" "\\1\n" bytes "${bytes}")
get_filename_component(name "${INPUT}" NAME)
file(WRITE "${OUTPUT}"
  "// Generated from ${name} by cmake/embed.cmake - do not edit.\n"
  "#include <cstddef>\n"
  "namespace ember::web {\n"
  "extern const unsigned char ${SYMBOL}[] = {\n${bytes}0};\n"
  "extern const size_t ${SYMBOL}_size = ${size};\n"
  "}\n")
