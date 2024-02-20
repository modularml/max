cmake_minimum_required(VERSION 3.24)

project(max-examples LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "$ENV{MAX_PKG_DIR}/lib/cmake")
include(AddMaxEngine)

add_executable(bert main.c)
target_link_libraries(bert PUBLIC max-engine)
