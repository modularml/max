cmake_minimum_required(VERSION 3.24)

project(max-examples LANGUAGES CXX C)

# TODO: Replace with exact package location
list(APPEND CMAKE_MODULE_PATH "$ENV{MODULAR_PKG_HOME}/lib/cmake")
include(AddMaxEngine)

add_executable(bert main.c)
target_link_libraries(bert PUBLIC max-engine)
