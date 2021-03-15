foreach(TargetName miscellaneous_ex5 cahn_hilliard_ex three_species)
    add_executable(${TargetName} ${TargetName}.cpp)
    target_link_libraries(${TargetName} CahnHilliard)
    target_link_libraries(${TargetName} ${VTK_LIBRARIES})

    # create output directory
    add_custom_command(
            TARGET ${TargetName} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TargetName}>/output)

    # link config file
    add_custom_command(
            TARGET ${TargetName} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/${TargetName}.in $<TARGET_FILE_DIR:${TargetName}>/${TargetName}.in
            COMMENT "Linking config file: ${CMAKE_CURRENT_SOURCE_DIR}/${TargetName}.in")
endforeach()
