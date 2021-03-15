set(SOURCES
        chdg/integrator.cpp
        chdg/time_measurement.cpp
        chdg/forms.cpp
        chdg/cahn_hilliard.cpp
)

set(HEADERS
        chdg/integrator.h
        chdg/mark_boundary.h
        chdg/time_measurement.h
        chdg/forms.h
        chdg/assembly_data.h
        chdg/local_system.h
        chdg/cahn_hilliard.h
)

add_library(CahnHilliard ${SOURCES} ${HEADERS})
target_include_directories(CahnHilliard INTERFACE ./)
target_include_directories(CahnHilliard PUBLIC ${MPI_INCLUDE_PATH})
target_include_directories(CahnHilliard PUBLIC ${LIBMESH_INCLUDE_DIRS})
target_link_libraries(CahnHilliard PUBLIC ${LIBMESH_LIBRARIES})
target_link_libraries(CahnHilliard PUBLIC ${MPI_LIBRARIES})
target_link_libraries(CahnHilliard PUBLIC Threads::Threads)
