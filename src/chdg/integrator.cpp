#include "integrator.h"

#include "assembly_data.h"
#include "forms.h"
#include "libmesh/boundary_info.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/fe_map.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/sparse_matrix.h"
#include "local_system.h"
#include "time_measurement.h"

namespace chdg {
namespace integrator {

class DGFaceAssemblyUtil
{
 public:
   DGFaceAssemblyUtil( const unsigned int dim, const lm::FEType& fe_type_trial, const lm::FEType& fe_type_test )
   : fe_e( lm::FEBase::build( dim, fe_type_trial ) )
   , fe_n( lm::FEBase::build( dim, fe_type_trial ) )
   , qrule_( dim - 1, fe_type_trial.default_quadrature_order() )
   , is_exterior_( false )
   {
      if ( fe_type_trial != fe_type_test )
         throw std::runtime_error( "Different elements for trial and test functions not supported yet." );

      fe_e->attach_quadrature_rule( &qrule_ );
      fe_n->attach_quadrature_rule( &qrule_ );
   }

   /// Computes the values of the shape functions for a face and its neighbor.
   void init( const lm::Elem* elem, const unsigned short side )
   {
      is_exterior_ = elem->neighbor_ptr( side ) == nullptr;
      if ( is_exterior_ )
      {
         init_exterior( elem, side );
      }
      else
      {
         init_interior( elem, side );
      }
   }

   const std::vector< std::vector< lm::Real > >& get_phi_e() const { return fe_e->get_phi(); }

   const std::vector< std::vector< lm::RealGradient > >& get_dphi_e() const { return fe_e->get_dphi(); }

   const std::vector< lm::Real >& get_JxW_e() const { return fe_e->get_JxW(); }

   const std::vector< lm::Point >& get_normals_e() const { return fe_e->get_normals(); }

   const std::vector< lm::Point >& get_xyz_e() const { return fe_e->get_xyz(); }

   const std::vector< std::vector< lm::Real > >& get_phi_n() const { return fe_n->get_phi(); }

   const std::vector< std::vector< lm::RealGradient > >& get_dphi_n() const { return fe_n->get_dphi(); }

   const std::vector< std::vector< lm::Real > >& get_psi_e() const { return fe_e->get_phi(); }

   const std::vector< std::vector< lm::RealGradient > >& get_dpsi_e() const { return fe_e->get_dphi(); }

   const std::vector< std::vector< lm::Real > >& get_psi_n() const { return fe_n->get_phi(); }

   const std::vector< std::vector< lm::RealGradient > >& get_dpsi_n() const { return fe_n->get_dphi(); }

   bool is_exterior() const { return is_exterior_; }

   const lm::QBase& get_qrule() const { return qrule_; };

   unsigned int get_order_e() const { return fe_e->get_order(); }

   unsigned int get_order_n() const { return fe_n->get_order(); }

 private:
   void init_exterior( const lm::Elem* elem, const unsigned short side ) { fe_e->reinit( elem, side ); }

   void init_interior( const lm::Elem* elem, const unsigned short side )
   {
      // initialize the element itself
      fe_e->reinit( elem, side );

      const lm::Elem* neighbor = elem->neighbor_ptr( side );

      // we now map the quadrature points elem to its neighbor
      auto qrule_face_points = fe_e->get_xyz();
      std::vector< lm::Point > qrule_face_neighbor_points;
      lm::FEMap::inverse_map( elem->dim(), neighbor, qrule_face_points, qrule_face_neighbor_points );
      fe_n->reinit( neighbor, &qrule_face_neighbor_points );
   }

 private:
   std::unique_ptr< lm::FEBase > fe_e;
   std::unique_ptr< lm::FEBase > fe_n;

   lm::QGauss qrule_;

   bool is_exterior_;
};

EllipticIntegrator::EllipticIntegrator( lm::LinearImplicitSystem& sys )
: sys_( sys )
, trial_function_( 0 )
, test_function_( 0 )
, penalty_( 10. )
, eps_( -1 )
, neumann_boundary_value_( []( const auto&, auto& values ) { std::fill( values.begin(), values.end(), 0 ); } )
, dirichlet_boundary_value_( []( const auto&, auto& values ) { std::fill( values.begin(), values.end(), 0 ); } )
, kappa_( []( const auto&, const auto&, auto& values ) { std::fill( values.begin(), values.end(), 1 ); } )
, dirichlet_boundary_id_( 0 )
, tau_( 1. )
{}

void EllipticIntegrator::assemble()
{
   auto& K = sys_.get_system_matrix();
   auto& F = *sys_.rhs;

   assemble( K, F );
}

void EllipticIntegrator::assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const
{
   // TimeMeasurement tm( "assembly elliptic operator" );
   assemble_elements( K );
   assemble_faces( K, F );
}

void EllipticIntegrator::assemble_elements( lm::SparseMatrix< lm::Real >& K ) const
{
   lm::out << "assembling elliptic dg system on elements... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   if ( fe_type_trial != fe_type_test )
      throw std::runtime_error( "different finite-element types for trial and test functions not supported yet." );

   LocalCellSystem lcs;
   CellAssemblyData cell_assembly_data( dim, fe_type_trial, fe_type_test );

   std::vector< lm::Real > kappa_values;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      lcs.init( dof_map, trial_function_, test_function_, *elem );
      cell_assembly_data.init( elem );

      kappa_values.resize( cell_assembly_data.num_quad_points() );
      kappa_( elem, cell_assembly_data.xyz, kappa_values );

      forms::diffusion::assemble_cell( lcs, cell_assembly_data, kappa_values, tau_ );

      K.add_matrix( lcs.K, lcs.dof_indices_psi, lcs.dof_indices_phi );
   }

   lm::out << "done" << std::endl;
}

void EllipticIntegrator::assemble_faces( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const
{
   lm::out << "assembling elliptic dg system on faces... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const double beta_ = 1. / ( dim - 1 );

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   DGFaceAssemblyUtil dg_util( dim, fe_type_trial, fe_type_test );

   const auto& phi_e = dg_util.get_phi_e();
   const auto& dphi_e = dg_util.get_dphi_e();
   const auto& psi_e = dg_util.get_psi_e();
   const auto& dpsi_e = dg_util.get_dpsi_e();

   const auto& phi_n = dg_util.get_phi_n();
   const auto& dphi_n = dg_util.get_dphi_n();
   const auto& psi_n = dg_util.get_psi_n();
   const auto& dpsi_n = dg_util.get_dpsi_n();

   const auto& JxW = dg_util.get_JxW_e();
   const auto& normals_e = dg_util.get_normals_e();
   const auto& quad_points_e = dg_util.get_xyz_e();

   LocalCellSystem lcs;
   LocalInteriorFaceSystem lifs;

   FaceAssemblyData face_assembly_data( dim, fe_type_trial, fe_type_test );

   std::vector< lm::Real > dbc_values;
   std::vector< lm::Real > nbc_values;

   std::vector< lm::Real > kappa_values_e;
   std::vector< lm::Real > kappa_values_n;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      for ( auto side : elem->side_index_range() )
      {
         dg_util.init( elem, side );
         face_assembly_data.init( elem, side );

         if ( dg_util.is_exterior() )
         {
            lcs.init( dof_map, trial_function_, test_function_, *elem );

            kappa_values_e.resize( dg_util.get_qrule().n_points() );
            kappa_( elem, quad_points_e, kappa_values_e );

            std::unique_ptr< const lm::Elem > elem_side( elem->build_side_ptr( side ) );
            const lm::Real h_elem = pow( elem_side->volume(), beta_ );

            for ( unsigned int i = 0; i < lcs.dof_indices_psi.size(); i += 1 )
            {
               // matrix contributions:
               for ( unsigned int j = 0; j < lcs.dof_indices_phi.size(); j += 1 )
               {
                  for ( unsigned int qp = 0; qp < dg_util.get_qrule().n_points(); qp += 1 )
                  {
                     // stability
                     lcs.K( i, j ) += tau_ * JxW[qp] * penalty_ / h_elem * psi_e[i][qp] * phi_e[j][qp];

                     // consistency
                     lcs.K( i, j ) += tau_ * JxW[qp] * kappa_values_e[qp] *
                                      ( -psi_e[i][qp] * ( dphi_e[j][qp] * normals_e[qp] ) +
                                        eps_ * phi_e[j][qp] * ( dpsi_e[i][qp] * normals_e[qp] ) );
                  }
               }

               // rhs contributions:
               {
                  const bool isDirichletBdry = mesh.get_boundary_info().has_boundary_id( elem, side, dirichlet_boundary_id_ );

                  if ( isDirichletBdry )
                  {
                     dbc_values.resize( dg_util.get_qrule().n_points(), 0 );
                     dirichlet_boundary_value_( quad_points_e, dbc_values );

                     for ( unsigned int qp = 0; qp < dg_util.get_qrule().n_points(); qp += 1 )
                     {
                        // stability
                        lcs.F( i ) += tau_ * JxW[qp] * dbc_values[qp] * penalty_ / h_elem * psi_e[i][qp];

                        // consistency
                        lcs.F( i ) +=
                            tau_ * eps_ * JxW[qp] * kappa_values_e[qp] * dpsi_e[i][qp] * ( dbc_values[qp] * normals_e[qp] );
                     }
                  }
                  // if it is not a dirichlet boundary, then it must be a neumann boundary
                  else
                  {
                     nbc_values.resize( dg_util.get_qrule().n_points(), 0 );
                     neumann_boundary_value_( quad_points_e, nbc_values );

                     for ( unsigned int qp = 0; qp < dg_util.get_qrule().n_points(); qp += 1 )
                        lcs.F( i ) += tau_ * JxW[qp] * kappa_values_e[qp] * psi_e[i][qp] * nbc_values[qp];
                  }
               }
            }

            K.add_matrix( lcs.K, lcs.dof_indices_psi, lcs.dof_indices_phi );
            F.add_vector( lcs.F, lcs.dof_indices_psi );
         }
         else
         {
            const lm::Elem* neighbor = elem->neighbor_ptr( side );

            const unsigned int elem_id = elem->id();
            const unsigned int neighbor_id = neighbor->id();

            if ( ( neighbor->active() && ( neighbor->level() == elem->level() ) && ( elem_id < neighbor_id ) ) ||
                 ( neighbor->level() < elem->level() ) )
            {
               std::unique_ptr< const lm::Elem > elem_side( elem->build_side_ptr( side ) );

               kappa_values_e.resize( dg_util.get_qrule().n_points() );
               kappa_values_n.resize( dg_util.get_qrule().n_points() );
               kappa_( elem, quad_points_e, kappa_values_e );
               kappa_( neighbor, quad_points_e, kappa_values_n );

               lifs.init( dof_map, trial_function_, test_function_, *elem, *neighbor );

               forms::diffusion::assemble_face(
                   lifs, face_assembly_data, kappa_values_e, kappa_values_n, tau_, eps_, penalty_, beta_ );

               K.add_matrix( lifs.K, lifs.dof_indices_psi, lifs.dof_indices_phi );
            }
         }
      }
   }

   lm::out << "done" << std::endl;
}

AdvectionIntegrator::AdvectionIntegrator( lm::LinearImplicitSystem& sys )
: sys_( sys )
, trial_function_( 0 )
, test_function_( 0 )
, inflow_boundary_value_( []( const auto& ) { return 0.; } )
, velocity_field_( []( const auto& ) { return lm::Point{}; } )
, tau_( 1. )
{}

void AdvectionIntegrator::assemble()
{
   auto& K = sys_.get_system_matrix();
   auto& F = *sys_.rhs;

   assemble( K, F );
}

void AdvectionIntegrator::assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const
{
   assemble_elements( K );
   assemble_faces( K, F );
}

void AdvectionIntegrator::assemble_elements( lm::SparseMatrix< lm::Real >& K ) const
{
   lm::out << "assembling advective dg system on elements... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   if ( fe_type_trial != fe_type_test )
      throw std::runtime_error( "different finite-element types for trial and test functions not supported yet." );

   std::unique_ptr< lm::FEBase > fe( lm::FEBase::build( dim, fe_type_trial ) );
   lm::QGauss qrule( dim, fe_type_trial.default_quadrature_order() );
   fe->attach_quadrature_rule( &qrule );

   const auto& JxW = fe->get_JxW();
   const auto& phi = fe->get_phi();
   const auto& dphi = fe->get_dphi();
   const auto& psi = fe->get_phi();
   const auto& dpsi = fe->get_dphi();

   const auto& quad_points = fe->get_xyz();

   lm::DenseMatrix< lm::Number > Ke;

   std::vector< lm::dof_id_type > dof_indices_phi;
   std::vector< lm::dof_id_type > dof_indices_psi;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      dof_map.dof_indices( elem, dof_indices_phi, trial_function_ );
      dof_map.dof_indices( elem, dof_indices_psi, test_function_ );

      fe->reinit( elem );

      Ke.resize( dof_indices_psi.size(), dof_indices_phi.size() );

      for ( unsigned int qp = 0; qp < qrule.n_points(); qp += 1 )
      {
         const auto velocity = velocity_field_( quad_points[qp] );
         for ( unsigned int i = 0; i < dof_indices_psi.size(); i += 1 )
            for ( unsigned int j = 0; j < dof_indices_phi.size(); j += 1 )
               Ke( i, j ) += ( -tau_ ) * JxW[qp] * ( dpsi[i][qp] * velocity ) * phi[j][qp];
      }

      K.add_matrix( Ke, dof_indices_psi, dof_indices_phi );
   }

   lm::out << "done" << std::endl;
}

void AdvectionIntegrator::assemble_faces( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const
{
   lm::out << "assembling advective dg system on faces... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   FaceAssemblyData face_assembly_data( dim, fe_type_trial, fe_type_test );
   LocalInteriorFaceSystem lifs;
   LocalCellSystem lcs;

   DGFaceAssemblyUtil dg_util( dim, fe_type_trial, fe_type_test );

   const auto& phi_e = dg_util.get_phi_e();
   const auto& dphi_e = dg_util.get_dphi_e();
   const auto& psi_e = dg_util.get_psi_e();
   const auto& dpsi_e = dg_util.get_dpsi_e();
   const auto& JxW = dg_util.get_JxW_e();
   const auto& normals_e = dg_util.get_normals_e();
   const auto& quad_points_e = dg_util.get_xyz_e();

   const auto& phi_n = dg_util.get_phi_n();
   const auto& dphi_n = dg_util.get_dphi_n();
   const auto& psi_n = dg_util.get_psi_n();
   const auto& dpsi_n = dg_util.get_dpsi_n();

   std::vector< lm::Point > velocity_values;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      for ( auto side : elem->side_index_range() )
      {
         if ( elem->neighbor_ptr( side ) == nullptr )
         {
            dg_util.init( elem, side );
            lcs.init( dof_map, trial_function_, test_function_, *elem );

            for ( unsigned int qp = 0; qp < dg_util.get_qrule().n_points(); qp += 1 )
            {
               const auto velocity = velocity_field_( quad_points_e[qp] );
               const auto velocity_x_normal = velocity * normals_e[qp];

               const bool is_inflow_bdry = velocity_x_normal < 0;
               const auto inflow_value = inflow_boundary_value_( quad_points_e[qp] );

               for ( unsigned int i = 0; i < lcs.dof_indices_psi.size(); i += 1 )
               {
                  if ( is_inflow_bdry )
                  {
                     // rhs contributions:
                     {
                        lcs.F( i ) += ( -tau_ ) * JxW[qp] * inflow_value * velocity_x_normal * psi_e[i][qp];
                     }
                  }
                  else
                  {
                     // matrix contributions:
                     for ( unsigned int j = 0; j < lcs.dof_indices_phi.size(); j += 1 )
                     {
                        lcs.K( i, j ) += tau_ * JxW[qp] * psi_e[i][qp] * velocity_x_normal * phi_e[j][qp];
                     }
                  }
               }
            }

            K.add_matrix( lcs.K, lcs.dof_indices_psi, lcs.dof_indices_phi );
            F.add_vector( lcs.F, lcs.dof_indices_psi );
         }
         else
         {
            const lm::Elem* neighbor = elem->neighbor_ptr( side );

            const unsigned int elem_id = elem->id();
            const unsigned int neighbor_id = neighbor->id();

            if ( ( neighbor->active() && ( neighbor->level() == elem->level() ) && ( elem_id < neighbor_id ) ) ||
                 ( neighbor->level() < elem->level() ) )
            {
               lifs.init( dof_map, trial_function_, test_function_, *elem, *neighbor );
               face_assembly_data.init( elem, side );

               velocity_values.resize( face_assembly_data.num_quad_points() );
               for ( unsigned int qp = 0; qp < face_assembly_data.num_quad_points(); qp += 1 )
                  velocity_values[qp] = velocity_field_( face_assembly_data.xyz_e[qp] );

               forms::advection::assemble_face( lifs, face_assembly_data, velocity_values, tau_ );

               K.add_matrix( lifs.K, lifs.dof_indices_psi, lifs.dof_indices_phi );
            }
         }
      }
   }

   lm::out << "done" << std::endl;
}

MassLFIntegrator::MassLFIntegrator( lm::LinearImplicitSystem& sys, lm::NumericVector< lm::Real >& u )
: sys_( sys )
, trial_function_( 0 )
, test_function_( 0 )
, u_( u )
{}

void MassLFIntegrator::assemble()
{
   auto& F = *sys_.rhs;
   assemble( F );
}

void MassLFIntegrator::assemble( lm::NumericVector< lm::Real >& F ) const
{
   lm::out << "assembling mass rhs on elements... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   if ( fe_type_trial != fe_type_test )
      throw std::runtime_error( "different finite-element types for trial and test functions not supported yet." );

   std::unique_ptr< lm::FEBase > fe( lm::FEBase::build( dim, fe_type_trial ) );
   lm::QGauss qrule( dim, fe_type_trial.default_quadrature_order() );
   fe->attach_quadrature_rule( &qrule );

   const auto& JxW = fe->get_JxW();
   const auto& phi = fe->get_phi();
   const auto& psi = fe->get_phi();

   lm::DenseVector< lm::Number > Fe;

   std::vector< lm::dof_id_type > dof_indices_phi;
   std::vector< lm::dof_id_type > dof_indices_psi;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      dof_map.dof_indices( elem, dof_indices_phi, trial_function_ );
      dof_map.dof_indices( elem, dof_indices_psi, test_function_ );

      fe->reinit( elem );

      Fe.resize( dof_indices_psi.size() );

      for ( unsigned int qp = 0; qp < qrule.n_points(); qp += 1 )
      {
         for ( unsigned int i = 0; i < dof_indices_psi.size(); i += 1 )
            for ( unsigned int j = 0; j < dof_indices_phi.size(); j += 1 )
               Fe( i ) += JxW[qp] * psi[i][qp] * phi[j][qp] * u_( dof_indices_phi[j] );
      }

      F.add_vector( Fe, dof_indices_psi );
   }

   lm::out << "done" << std::endl;
}

MassIntegrator::MassIntegrator( lm::LinearImplicitSystem& sys )
: sys_( sys )
, trial_function_( 0 )
, test_function_( 0 )
{}

void MassIntegrator::assemble()
{
   auto& K = sys_.get_system_matrix();
   assemble( K );
}

void MassIntegrator::assemble( lm::SparseMatrix< lm::Real >& K ) const
{
   lm::out << "assembling mass matrix on elements... ";
   lm::out.flush();

   const lm::MeshBase& mesh = sys_.get_mesh();
   const unsigned int dim = mesh.mesh_dimension();

   const lm::DofMap& dof_map = sys_.get_dof_map();

   const lm::FEType fe_type_trial = sys_.variable_type( trial_function_ );
   const lm::FEType fe_type_test = sys_.variable_type( test_function_ );

   if ( fe_type_trial != fe_type_test )
      throw std::runtime_error( "different finite-element types for trial and test functions not supported yet." );

   std::unique_ptr< lm::FEBase > fe( lm::FEBase::build( dim, fe_type_trial ) );
   lm::QGauss qrule( dim, fe_type_trial.default_quadrature_order() );
   fe->attach_quadrature_rule( &qrule );

   const auto& JxW = fe->get_JxW();
   const auto& phi = fe->get_phi();
   const auto& psi = fe->get_phi();

   lm::DenseMatrix< lm::Number > Ke;

   std::vector< lm::dof_id_type > dof_indices_phi;
   std::vector< lm::dof_id_type > dof_indices_psi;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      dof_map.dof_indices( elem, dof_indices_phi, trial_function_ );
      dof_map.dof_indices( elem, dof_indices_psi, test_function_ );

      fe->reinit( elem );

      Ke.resize( dof_indices_psi.size(), dof_indices_phi.size() );

      for ( unsigned int qp = 0; qp < qrule.n_points(); qp += 1 )
      {
         for ( unsigned int i = 0; i < dof_indices_psi.size(); i += 1 )
            for ( unsigned int j = 0; j < dof_indices_phi.size(); j += 1 )
               Ke( i, j ) += JxW[qp] * psi[i][qp] * phi[j][qp];
      }

      K.add_matrix( Ke, dof_indices_psi, dof_indices_phi );
   }

   lm::out << "done" << std::endl;
}

TransientImplicitAdvectionIntegrator::TransientImplicitAdvectionIntegrator( lm::TransientLinearImplicitSystem& sys )
: sys_( sys )
, advection_( sys )
, mass_( sys )
, mass_lf_( sys, *sys.old_local_solution )
{}

void TransientImplicitAdvectionIntegrator::set_trial_function( unsigned int trial_function )
{
   advection_.set_trial_function( trial_function );
   mass_.set_trial_function( trial_function );
   mass_lf_.set_trial_function( trial_function );
}

void TransientImplicitAdvectionIntegrator::set_test_function( unsigned int test_function )
{
   mass_.set_test_function( test_function );
   mass_lf_.set_test_function( test_function );
   advection_.set_test_function( test_function );
}

void TransientImplicitAdvectionIntegrator::set_dt( double dt )
{
   advection_.set_tau( dt );
}

void TransientImplicitAdvectionIntegrator::assemble()
{
   auto& K = sys_.get_system_matrix();
   auto& F = *sys_.rhs;

   assemble( K, F );
}

void TransientImplicitAdvectionIntegrator::assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const
{
   advection_.assemble( K, F );
   mass_.assemble( K );
   mass_lf_.assemble( F );
}

} // namespace integrator
} // namespace chdg