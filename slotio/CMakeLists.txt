cmake_minimum_required(VERSION 3.26)
project(slotio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O2 -Wall -Wextra)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(slotio_core STATIC src/reader.cpp src/pool.cpp src/clock.cpp src/ensure.cpp src/rows.cpp src/embed.cpp src/wait.cpp src/ledger.cpp src/arbiter.cpp src/sweep.cpp src/prefetch.cpp src/control_ring.cpp)
target_link_libraries(slotio_core PUBLIC ${CMAKE_DL_LIBS})
target_include_directories(slotio_core PUBLIC include)
target_link_libraries(slotio_core PUBLIC Threads::Threads CUDA::cuda_driver)
target_include_directories(slotio_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

nanobind_add_module(slotio NB_STATIC src/bindings.cpp)
target_link_libraries(slotio PRIVATE slotio_core)
set_target_properties(slotio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

enable_testing()
add_executable(slotio_tests tests/test_main.cpp tests/test_reader.cpp tests/test_pool.cpp tests/test_ensure.cpp tests/test_rows.cpp tests/test_arbiter.cpp tests/test_sweep.cpp tests/test_prefetch.cpp tests/test_control_ring.cpp)
target_link_libraries(slotio_tests PRIVATE slotio_core)
target_compile_definitions(slotio_tests PRIVATE
  SLOTIO_BUILD_DIR="${CMAKE_BINARY_DIR}" SLOTIO_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME slotio COMMAND slotio_tests)
