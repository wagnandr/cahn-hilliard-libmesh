#pragma once

#include "libmesh/dense_matrix.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"

namespace chdg {

namespace lm = libMesh;

struct LocalCellSystem
{
   lm::DenseMatrix< lm::Number > K;
   lm::DenseVector< lm::Number > F;

   std::vector< lm::dof_id_type > dof_indices_phi;
   std::vector< lm::dof_id_type > dof_indices_psi;

   void init( const lm::DofMap& dofmap,
              const unsigned int trial_function,
              const unsigned int test_function,
              const lm::Elem& element )
   {
      dofmap.dof_indices( &element, dof_indices_phi, trial_function );
      dofmap.dof_indices( &element, dof_indices_psi, test_function );

      K.resize( dof_indices_psi.size(), dof_indices_phi.size() );
      F.resize( dof_indices_psi.size() );
   }
};

class LocalInteriorFaceSystem
{
 public:
   explicit LocalInteriorFaceSystem()
   : Kee( K )
   , Ken( K )
   , Kne( K )
   , Knn( K ){};

   lm::DenseMatrix< lm::Number > K;

   lm::DenseSubMatrix< lm::Number > Kee;
   lm::DenseSubMatrix< lm::Number > Ken;
   lm::DenseSubMatrix< lm::Number > Kne;
   lm::DenseSubMatrix< lm::Number > Knn;

   std::vector< lm::dof_id_type > dof_indices_phi;
   std::vector< lm::dof_id_type > dof_indices_psi;

   std::vector< lm::dof_id_type > dof_indices_phi_e;
   std::vector< lm::dof_id_type > dof_indices_psi_e;

   std::vector< lm::dof_id_type > dof_indices_phi_n;
   std::vector< lm::dof_id_type > dof_indices_psi_n;

   void init( const lm::DofMap& dofmap,
              const unsigned int trial_function,
              const unsigned int test_function,
              const lm::Elem& element,
              const lm::Elem& neighbor )
   {
      dofmap.dof_indices( &element, dof_indices_phi_e, trial_function );
      dofmap.dof_indices( &element, dof_indices_psi_e, test_function );

      dofmap.dof_indices( &neighbor, dof_indices_phi_n, trial_function );
      dofmap.dof_indices( &neighbor, dof_indices_psi_n, test_function );

      const auto n_dofs_psi = dof_indices_psi_e.size() + dof_indices_psi_n.size();
      const auto n_dofs_phi = dof_indices_phi_e.size() + dof_indices_phi_n.size();

      K.resize( n_dofs_psi, n_dofs_phi );

      Kee.reposition( 0, 0, dof_indices_psi_e.size(), dof_indices_phi_e.size() );
      Ken.reposition( 0, dof_indices_phi_e.size(), dof_indices_psi_e.size(), dof_indices_phi_n.size() );
      Kne.reposition( dof_indices_psi_e.size(), 0, dof_indices_psi_n.size(), dof_indices_phi_e.size() );
      Knn.reposition( dof_indices_psi_e.size(), dof_indices_phi_e.size(), dof_indices_psi_n.size(), dof_indices_phi_n.size() );

      // copy into global dof vector
      dof_indices_phi.resize( 0 );
      dof_indices_phi.reserve( n_dofs_phi );
      dof_indices_phi.insert( dof_indices_phi.end(), dof_indices_phi_e.begin(), dof_indices_phi_e.end() );
      dof_indices_phi.insert( dof_indices_phi.end(), dof_indices_phi_n.begin(), dof_indices_phi_n.end() );

      // copy into global dof vector
      dof_indices_psi.resize( 0 );
      dof_indices_psi.reserve( n_dofs_psi );
      dof_indices_psi.insert( dof_indices_psi.end(), dof_indices_psi_e.begin(), dof_indices_psi_e.end() );
      dof_indices_psi.insert( dof_indices_psi.end(), dof_indices_psi_n.begin(), dof_indices_psi_n.end() );
   }
};

inline std::ostream& operator<<(std::ostream& os, const LocalInteriorFaceSystem &lifs)
{
   os << "dof_indices_phi ";
   for (std::size_t i =0; i < lifs.dof_indices_phi.size(); i+=1)
      os << lifs.dof_indices_phi[i] << " ";
   os << std::endl;

   os << "dof_indices_psi ";
   for (std::size_t i =0; i < lifs.dof_indices_psi.size(); i+=1)
      os << lifs.dof_indices_psi[i] << " ";
   os << std::endl;

   os << "dof_indices_phi_e ";
   for (std::size_t i =0; i < lifs.dof_indices_phi_e.size(); i+=1)
      os << lifs.dof_indices_phi_e[i] << " ";
   os << std::endl;

   os << "dof_indices_phi_n ";
   for (std::size_t i =0; i < lifs.dof_indices_phi_n.size(); i+=1)
      os << lifs.dof_indices_phi_n[i] << " ";
   os << std::endl;

   os << "dof_indices_psi_e ";
   for (std::size_t i =0; i < lifs.dof_indices_psi_e.size(); i+=1)
      os << lifs.dof_indices_psi_e[i] << " ";
   os << std::endl;

   os << "dof_indices_psi_n ";
   for (std::size_t i =0; i < lifs.dof_indices_psi_n.size(); i+=1)
      os << lifs.dof_indices_psi_n[i] << " ";
   os << std::endl;

   os << "K \n" <<  lifs.K << std::endl;
   os << "Kee \n" << lifs.Kee << std::endl;
   os << "Ken \n" << lifs.Ken << std::endl;
   os << "Kne \n" << lifs.Kne << std::endl;
   os << "Knn \n" << lifs.Knn << std::endl;

   return os;
}

} // namespace chdg