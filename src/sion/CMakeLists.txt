add_library(sion SHARED)

find_package(Torch REQUIRED)

target_sources(sion PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
    ${CMAKE_SOURCE_DIR}/include/sion/sion.hpp
)

target_link_libraries(sion PUBLIC torch felix)

add_subdirectory(flash_attention)
add_subdirectory(gemm)

install(TARGETS sion
    EXPORT SionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS
)

install(EXPORT SionTargets
    NAMESPACE Sion::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/SionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

