add_subdirectory(sgemm)

add_custom_target(bench EXCLUDE_FROM_ALL)
add_dependencies(bench
    sgemm_bench
)
