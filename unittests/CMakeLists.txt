# simple interface library for our test utils:
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE test_utils/)
target_link_libraries(test_utils INTERFACE Catch2::Catch2 CahnHilliard)

# test for elliptic operator
add_executable(test_elliptic src/test_elliptic.cpp)
target_link_libraries(test_elliptic test_utils)

add_test(
        NAME test_elliptic_catch
        COMMAND $<TARGET_FILE:test_elliptic> --success
)

# test for advection operator
add_executable(test_advection src/test_advection.cpp)
target_link_libraries(test_advection test_utils)

add_test(
        NAME test_advection_catch
        COMMAND $<TARGET_FILE:test_advection> --success
)
