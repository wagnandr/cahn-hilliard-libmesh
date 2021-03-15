#pragma once

#include <memory>

#include "libmesh/boundary_info.h"
#include "libmesh/elem.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"

namespace chdg {

namespace lm = libMesh;

template < typename BoundaryIndicatorType >
void mark_boundary( lm::Mesh& mesh, lm::boundary_id_type mark, BoundaryIndicatorType ind )
{
   for ( const auto& elem : mesh.element_ptr_range() )
   {
      for ( auto side : elem->side_index_range() )
      {
         if ( elem->neighbor_ptr( side ) == nullptr )
         {
            std::unique_ptr< const lm::Elem > elem_side( elem->build_side_ptr( side ) );
            if ( ind( elem_side->centroid() ) )
            {
               mesh.get_boundary_info().add_side( elem, side, mark );
            }
         }
      }
   }
}

} // namespace chdg
