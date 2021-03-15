#include "forms.h"

#include "assembly_data.h"
#include "local_system.h"

namespace chdg {

namespace forms {
namespace diffusion {

void assemble_cell( LocalCellSystem& ls, const CellAssemblyData& d, const std::vector< double >& kappa, const double& tau )
{
   for ( unsigned int i = 0; i < ls.dof_indices_psi.size(); i += 1 )
      for ( unsigned int j = 0; j < ls.dof_indices_phi.size(); j += 1 )
         for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
            ls.K( i, j ) += tau * d.JxW[qp] * kappa[qp] * ( d.dpsi[i][qp] * d.dphi[j][qp] );
}

void assemble_face( LocalInteriorFaceSystem& ls,
                    const FaceAssemblyData& d,
                    const std::vector< double >& kappa_e,
                    const std::vector< double >& kappa_n,
                    const double& tau,
                    const double& eps,
                    const double& penalty,
                    const double& beta )
{
   const auto h_elem = std::pow( d.get_h(), beta );

   for ( unsigned int i = 0; i < d.num_shape_functions_e(); i += 1 )
   {
      for ( unsigned int j = 0; j < d.num_shape_functions_e(); j += 1 )
      {
         for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
         {
            // consistency
            ls.Kee( i, j ) += tau * 0.5 * d.JxW[qp] *
                              ( +eps * d.phi_e[j][qp] * ( d.n_e[qp] * d.dpsi_e[i][qp] * kappa_e[qp] ) -
                                d.psi_e[i][qp] * ( d.n_e[qp] * d.dphi_e[j][qp] * kappa_e[qp] ) );

            // stability
            ls.Kee( i, j ) += tau * d.JxW[qp] * penalty / h_elem * d.phi_e[j][qp] * d.psi_e[i][qp];
         }
      }
   }

   // Knn matrix: integrate the neighbor test function i against the neighbor test function j
   for ( unsigned int i = 0; i < d.num_shape_functions_n(); i += 1 )
   {
      for ( unsigned int j = 0; j < d.num_shape_functions_n(); j += 1 )
      {
         for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
         {
            // consistency
            ls.Knn( i, j ) += tau * 0.5 * d.JxW[qp] *
                              ( -eps * d.phi_n[j][qp] * ( d.n_e[qp] * d.dpsi_n[i][qp] * kappa_n[qp] ) +
                                d.psi_n[i][qp] * ( d.n_e[qp] * d.dphi_n[j][qp] * kappa_n[qp] ) );

            // stability
            ls.Knn( i, j ) += tau * d.JxW[qp] * penalty / h_elem * d.phi_n[j][qp] * d.psi_n[i][qp];
         }
      }
   }

   // Kne matrix: integrate the neighbor test function i against the element test function j
   for ( unsigned int i = 0; i < d.num_shape_functions_n(); i += 1 )
   {
      for ( unsigned int j = 0; j < d.num_shape_functions_e(); j += 1 )
      {
         for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
         {
            // consistency
            ls.Kne( i, j ) += tau * 0.5 * d.JxW[qp] *
                              ( d.psi_n[i][qp] * ( d.n_e[qp] * d.dphi_e[j][qp] * kappa_e[qp] ) +
                                eps * d.phi_e[j][qp] * ( d.n_e[qp] * d.dpsi_n[i][qp] * kappa_n[qp] ) );

            // stability
            ls.Kne( i, j ) -= tau * d.JxW[qp] * penalty / h_elem * d.phi_e[j][qp] * d.psi_n[i][qp];
         }
      }
   }

   // Ken matrix: integrate the element test function i against the neighbor test function j
   for ( unsigned int i = 0; i < d.num_shape_functions_e(); i += 1 )
   {
      for ( unsigned int j = 0; j < d.num_shape_functions_n(); j += 1 )
      {
         for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
         {
            // consistency
            ls.Ken( i, j ) += tau * 0.5 * d.JxW[qp] *
                              ( -eps * d.phi_n[j][qp] * ( d.n_e[qp] * d.dpsi_e[i][qp] * kappa_e[qp] ) -
                                d.psi_e[i][qp] * ( d.n_e[qp] * d.dphi_n[j][qp] * kappa_n[qp] ) );

            // stability
            ls.Ken( i, j ) -= tau * d.JxW[qp] * penalty / h_elem * d.psi_e[i][qp] * d.phi_n[j][qp];
         }
      }
   }
}

} // namespace diffusion

namespace advection {

void assemble_face( LocalInteriorFaceSystem& local_system,
                    const FaceAssemblyData& d,
                    const std::vector< lm::Point >& velocity,
                    const double& tau )
{
   auto& ls = local_system;
   // the test functions (i) against the trial functions (j).
   for ( unsigned int qp = 0; qp < d.num_quad_points(); qp += 1 )
   {
      const auto velocity_x_normal = velocity[qp] * d.n_e[qp];

      // Kee matrix: integrate the element test function i against the element test function j
      for ( unsigned int i = 0; i < d.psi_e.size(); i += 1 )
         for ( unsigned int j = 0; j < d.phi_e.size(); j += 1 )
            if ( velocity[qp] * d.n_e[qp] > 0 )
               ls.Kee( i, j ) += tau * d.JxW[qp] * d.phi_e[j][qp] * velocity_x_normal * d.psi_e[i][qp];

      // Knn matrix: integrate the neighbor test function i against the neighbor test function j
      for ( unsigned int i = 0; i < d.psi_n.size(); i += 1 )
         for ( unsigned int j = 0; j < d.phi_n.size(); j += 1 )
            if ( velocity[qp] * d.n_e[qp] < 0 )
               ls.Knn( i, j ) += ( -tau ) * d.JxW[qp] * d.phi_n[j][qp] * velocity_x_normal * d.psi_n[i][qp];

      // Kne matrix: integrate the neighbor test function i against the element test function j
      for ( unsigned int i = 0; i < d.psi_n.size(); i += 1 )
         for ( unsigned int j = 0; j < d.phi_e.size(); j += 1 )
            if ( velocity[qp] * d.n_e[qp] > 0 )
               ls.Kne( i, j ) += ( -tau ) * d.JxW[qp] * d.phi_e[j][qp] * velocity_x_normal * d.psi_n[i][qp];

      // Ken matrix: integrate the element test function i against the neighbor test function j
      for ( unsigned int i = 0; i < d.psi_e.size(); i += 1 )
         for ( unsigned int j = 0; j < d.phi_n.size(); j += 1 )
            if ( velocity[qp] * d.n_e[qp] < 0 )
               ls.Ken( i, j ) += tau * d.JxW[qp] * d.phi_n[j][qp] * velocity_x_normal * d.psi_e[i][qp];
   }
}

} // namespace advection

} // namespace forms
} // namespace chdg
