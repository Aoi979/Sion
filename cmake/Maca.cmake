include_guard(GLOBAL)

set(_cuda_ops_maca_default_path "/opt/maca")
if(DEFINED ENV{MACA_PATH} AND NOT "$ENV{MACA_PATH}" STREQUAL "")
    set(_cuda_ops_maca_default_path "$ENV{MACA_PATH}")
endif()

set(CUDA_OPS_MACA_PATH
    "${_cuda_ops_maca_default_path}"
    CACHE PATH "MACA installation prefix")

if(DEFINED ENV{MXCC} AND NOT "$ENV{MXCC}" STREQUAL "")
    set(_cuda_ops_maca_default_compiler "$ENV{MXCC}")
else()
    set(_cuda_ops_maca_default_compiler
        "${CUDA_OPS_MACA_PATH}/mxgpu_llvm/bin/mxcc")
endif()

set(CUDA_OPS_MACA_COMPILER
    "${_cuda_ops_maca_default_compiler}"
    CACHE FILEPATH "MACA mxcc compiler")

if(IS_ABSOLUTE "${CUDA_OPS_MACA_COMPILER}")
    set(_cuda_ops_maca_compiler "${CUDA_OPS_MACA_COMPILER}")
else()
    find_program(_cuda_ops_maca_compiler
        NAMES "${CUDA_OPS_MACA_COMPILER}"
        HINTS "${CUDA_OPS_MACA_PATH}/mxgpu_llvm/bin")
endif()

if(NOT _cuda_ops_maca_compiler OR NOT EXISTS "${_cuda_ops_maca_compiler}")
    message(FATAL_ERROR
        "MACA compiler not found: ${CUDA_OPS_MACA_COMPILER}")
endif()
set(CUDA_OPS_MACA_COMPILER
    "${_cuda_ops_maca_compiler}"
    CACHE FILEPATH "MACA mxcc compiler"
    FORCE)

set(CUDA_OPS_MACA_INCLUDE_DIR
    "${CUDA_OPS_MACA_PATH}/include"
    CACHE PATH "MACA include directory")
set(CUDA_OPS_MACA_COMMON_INCLUDE_DIR
    "${CUDA_OPS_MACA_INCLUDE_DIR}/common"
    CACHE PATH "MACA common include directory")
set(CUDA_OPS_MACA_MCR_INCLUDE_DIR
    "${CUDA_OPS_MACA_INCLUDE_DIR}/mcr"
    CACHE PATH "MACA runtime include directory")

foreach(_cuda_ops_maca_include_dir
    "${CUDA_OPS_MACA_INCLUDE_DIR}"
    "${CUDA_OPS_MACA_COMMON_INCLUDE_DIR}"
    "${CUDA_OPS_MACA_MCR_INCLUDE_DIR}")
    if(NOT IS_DIRECTORY "${_cuda_ops_maca_include_dir}")
        message(FATAL_ERROR
            "MACA include directory not found: ${_cuda_ops_maca_include_dir}")
    endif()
endforeach()

set(CUDA_OPS_MACA_CXX_STANDARD
    17
    CACHE STRING "C++ standard passed to mxcc")
set(CUDA_OPS_MACA_OFFLOAD_ARCH
    native
    CACHE STRING "MACA offload architecture passed to mxcc")
set(CUDA_OPS_MACA_COMPILE_OPTIONS
    ""
    CACHE STRING "Extra semicolon-separated options passed to mxcc")

function(cuda_ops_add_maca_object target)
    cmake_parse_arguments(
        MACA
        ""
        "SOURCE;OUTPUT"
        "DEPENDS"
        ${ARGN})

    if(NOT MACA_SOURCE)
        message(FATAL_ERROR
            "cuda_ops_add_maca_object requires SOURCE")
    endif()

    if(IS_ABSOLUTE "${MACA_SOURCE}")
        set(_cuda_ops_maca_source "${MACA_SOURCE}")
    else()
        set(_cuda_ops_maca_source
            "${CMAKE_CURRENT_SOURCE_DIR}/${MACA_SOURCE}")
    endif()

    if(NOT MACA_OUTPUT)
        set(_cuda_ops_maca_output
            "${CMAKE_CURRENT_BINARY_DIR}/${target}.o")
    elseif(IS_ABSOLUTE "${MACA_OUTPUT}")
        set(_cuda_ops_maca_output "${MACA_OUTPUT}")
    else()
        set(_cuda_ops_maca_output
            "${CMAKE_CURRENT_BINARY_DIR}/${MACA_OUTPUT}")
    endif()

    get_filename_component(_cuda_ops_maca_output_dir
        "${_cuda_ops_maca_output}" DIRECTORY)

    add_custom_command(
        OUTPUT "${_cuda_ops_maca_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${_cuda_ops_maca_output_dir}"
        COMMAND "${CUDA_OPS_MACA_COMPILER}"
            -x maca
            -O3
            "-std=c++${CUDA_OPS_MACA_CXX_STANDARD}"
            -offload-arch "${CUDA_OPS_MACA_OFFLOAD_ARCH}"
            "-I${PROJECT_SOURCE_DIR}"
            "-I${CUDA_OPS_MACA_INCLUDE_DIR}"
            "-I${CUDA_OPS_MACA_COMMON_INCLUDE_DIR}"
            "-I${CUDA_OPS_MACA_MCR_INCLUDE_DIR}"
            ${CUDA_OPS_MACA_COMPILE_OPTIONS}
            -c "${_cuda_ops_maca_source}"
            -o "${_cuda_ops_maca_output}"
        DEPENDS
            "${_cuda_ops_maca_source}"
            ${MACA_DEPENDS}
        COMMENT "Building MACA object ${target}"
        COMMAND_EXPAND_LISTS
        VERBATIM)

    add_custom_target("${target}" ALL
        DEPENDS "${_cuda_ops_maca_output}")
endfunction()
