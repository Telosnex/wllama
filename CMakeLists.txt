cmake_minimum_required(VERSION 3.14)
project("wllama")

option(GGML_WEBGPU "Enable GGML WebGPU backend" ON)
option(GGML_WEBGPU_JSPI "Enable GGML WebGPU JSPI support" ON)
option(LLAMA_WASM_MEM64 "Enable 64-bit memory for WebAssembly builds" OFF)
option(WLLAMA_ENABLE_SERVER_CONTEXT_POC "Build proof-of-concept action that calls llama.cpp server_context directly" OFF)
if(WLLAMA_ENABLE_SERVER_CONTEXT_POC)
  set(LLAMA_HTTPLIB OFF CACHE BOOL "Disable httplib for wllama; browser downloads are handled in JS" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "Disable CURL for wllama" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama-server executable" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
endif()

set(WLLAMA_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp source directory")
if(NOT EXISTS "${WLLAMA_LLAMA_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "WLLAMA_LLAMA_CPP_DIR does not point to a llama.cpp checkout: ${WLLAMA_LLAMA_CPP_DIR}")
endif()
include("${WLLAMA_LLAMA_CPP_DIR}/cmake/build-info.cmake")

# Some vendored/cached llama.cpp checkouts do not have git metadata available.
# mtmd uses LLAMA_INSTALL_VERSION in set_target_properties(VERSION ...), so
# provide the same fallback fllama uses for copied source trees.
if(NOT DEFINED LLAMA_INSTALL_VERSION OR LLAMA_INSTALL_VERSION STREQUAL "")
  set(LLAMA_INSTALL_VERSION "0.0.0")
endif()

add_subdirectory("${WLLAMA_LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
if(NOT DEFINED LLAMA_BUILD_NUMBER OR LLAMA_BUILD_NUMBER STREQUAL "")
  set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
endif()
if(NOT DEFINED LLAMA_BUILD_COMMIT OR LLAMA_BUILD_COMMIT STREQUAL "")
  set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
endif()

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(WLLAMA_EXTRA_LIBS ggml llama ${CMAKE_THREAD_LIBS_INIT})
if(WLLAMA_ENABLE_SERVER_CONTEXT_POC)
  add_subdirectory("${WLLAMA_LLAMA_CPP_DIR}/vendor/cpp-httplib" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-cpp-httplib")
  add_subdirectory("${WLLAMA_LLAMA_CPP_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-common")
  add_subdirectory("${WLLAMA_LLAMA_CPP_DIR}/tools/mtmd" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-mtmd")

  # Reuse the same non-HTTP server core fllama links natively.
  # Do not add_subdirectory(tools/server): that also builds llama-server and
  # pulls httplib/HTTP concerns that are irrelevant to this POC.
  set(WLLAMA_SERVER_SRC_DIR "${WLLAMA_LLAMA_CPP_DIR}/tools/server")
  add_library(wllama-server-context-poc STATIC
    ${WLLAMA_SERVER_SRC_DIR}/server-task.cpp
    ${WLLAMA_SERVER_SRC_DIR}/server-queue.cpp
    ${WLLAMA_SERVER_SRC_DIR}/server-common.cpp
    ${WLLAMA_SERVER_SRC_DIR}/server-chat.cpp
    ${WLLAMA_SERVER_SRC_DIR}/server-context.cpp
  )
  target_include_directories(wllama-server-context-poc PUBLIC ${WLLAMA_SERVER_SRC_DIR})
  target_include_directories(wllama-server-context-poc PRIVATE
    ${WLLAMA_LLAMA_CPP_DIR}/common
    ${WLLAMA_LLAMA_CPP_DIR}/tools/mtmd
    ${WLLAMA_LLAMA_CPP_DIR}/vendor
    ${WLLAMA_LLAMA_CPP_DIR}
  )
  target_compile_features(wllama-server-context-poc PRIVATE cxx_std_17)
  target_link_libraries(wllama-server-context-poc PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
  list(APPEND WLLAMA_EXTRA_LIBS llama-common cpp-httplib mtmd wllama-server-context-poc)
endif()

if(WLLAMA_ENABLE_SERVER_CONTEXT_POC)
  include_directories(${WLLAMA_LLAMA_CPP_DIR}/vendor)
endif()

set(WLLAMA_SRC cpp/wllama.cpp
    cpp/actions.hpp
    cpp/glue.hpp
    cpp/server_context_poc.cpp
    cpp/helpers/wlog.cpp
    cpp/helpers/wcommon.cpp
    cpp/helpers/wsampling.cpp
    ${WLLAMA_LLAMA_CPP_DIR}/include/llama.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/helpers)
include_directories(${WLLAMA_LLAMA_CPP_DIR}/include)
include_directories(${WLLAMA_LLAMA_CPP_DIR}/common)
include_directories(${WLLAMA_LLAMA_CPP_DIR}/tools/server)
include_directories(${WLLAMA_LLAMA_CPP_DIR})

add_executable(wllama ${WLLAMA_SRC})
target_compile_features(wllama PRIVATE cxx_std_17)
target_compile_definitions(wllama PRIVATE WLLAMA_ENABLE_SERVER_CONTEXT_POC=$<BOOL:${WLLAMA_ENABLE_SERVER_CONTEXT_POC}>)
target_link_libraries(wllama PRIVATE ${WLLAMA_EXTRA_LIBS})
