#include "cahn_hilliard.h"

#include <utility>

#include "assembly_data.h"
#include "forms.h"
#include "libmesh/dof_map.h"
#include "libmesh/getpot.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/transient_system.h"
#include "local_system.h"
#include "time_measurement.h"

namespace chdg {

CahnHilliardLocalSystemData::CahnHilliardLocalSystemData( const unsigned int variable_number_c,
                                                          const unsigned int variable_number_mu )
: K_cc( K )
, K_cmu( K )
, K_muc( K )
, K_mumu( K )
, F_c( F )
, F_mu( F )
, variable_number_c( variable_number_c )
, variable_number_mu( variable_number_mu )
{}

void CahnHilliardLocalSystemData::init( const lm::DofMap& dof_map, const lm::Elem& element )
{
   dof_map.dof_indices( &element, dof_indices );
   dof_map.dof_indices( &element, dof_indices_c, variable_number_c );
   dof_map.dof_indices( &element, dof_indices_mu, variable_number_mu );

   const auto size_c = dof_indices_c.size();
   const auto size_mu = dof_indices_mu.size();

   K.resize( size_c + size_mu, size_c + size_mu );
   K_cc.reposition( 0, 0, size_c, size_c );
   K_cmu.reposition( 0, size_c, size_c, size_mu );
   K_muc.reposition( size_c, 0, size_mu, size_c );
   K_mumu.reposition( size_c, size_c, size_mu, size_mu );

   F.resize( size_c + size_mu );
   F_c.reposition( 0, size_c );
   F_mu.reposition( size_c, size_mu );
}

EvaluatorOnQuadraturePoints::EvaluatorOnQuadraturePoints( const lm::TransientLinearImplicitSystem& sys, const std::string& name )
: sys_( sys )
, variable_number_( sys.variable_number( name ) )
{}

void EvaluatorOnQuadraturePoints::evaluate_old( const CellAssemblyData& data, std::vector< lm::Real >& values ) const
{
   sys_.get_dof_map().dof_indices( data.element, dof_indices, variable_number_ );
   sys_.old_local_solution->get( dof_indices, dof_values );
   values.resize( data.num_quad_points() );
   std::fill( values.begin(), values.end(), 0 );
   for ( unsigned int l = 0; l < data.phi.size(); l++ )
      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp++ )
         values[qp] += data.phi[l][qp] * dof_values[l];
}

void EvaluatorOnQuadraturePoints::evaluate_old( const FaceAssemblyData& data,
                                                std::vector< lm::Real >& values_e,
                                                std::vector< lm::Real >& values_n ) const
{
   sys_.get_dof_map().dof_indices( data.element, dof_indices_e, variable_number_ );
   sys_.get_dof_map().dof_indices( data.element, dof_indices_n, variable_number_ );

   sys_.old_local_solution->get( dof_indices_e, dof_values_e );
   sys_.old_local_solution->get( dof_indices_n, dof_values_n );

   values_e.resize( data.num_quad_points() );
   values_n.resize( data.num_quad_points() );

   std::fill( values_e.begin(), values_e.end(), 0 );
   std::fill( values_n.begin(), values_n.end(), 0 );

   for ( unsigned int l = 0; l < data.phi_e.size(); l++ )
      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp++ )
         values_e[qp] += data.phi_e[l][qp] * dof_values_e[l];
   for ( unsigned int l = 0; l < data.phi_n.size(); l++ )
      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp++ )
         values_n[qp] += data.phi_n[l][qp] * dof_values_n[l];
}

class DefaultEvaluateParametersOnCell : public EvaluateParametersOnCell
{
 public:
   explicit DefaultEvaluateParametersOnCell( const lm::TransientLinearImplicitSystem& sys ) {}
   ~DefaultEvaluateParametersOnCell() override = default;

   void init( const CellAssemblyData& ) override{};

   void mobility( std::vector< lm::Real >& mobility ) const override { std::fill( mobility.begin(), mobility.end(), 1 ); }

   void rhs_c( std::vector< lm::Real >& rhs ) const override { std::fill( rhs.begin(), rhs.end(), 0 ); }

   void rhs_mu( std::vector< lm::Real >& rhs ) const override { std::fill( rhs.begin(), rhs.end(), 0 ); }
};

class DefaultEvaluateParametersOnFace : public EvaluateParametersOnFace
{
 public:
   explicit DefaultEvaluateParametersOnFace( const lm::TransientLinearImplicitSystem& sys ) {}

   void init( const FaceAssemblyData& ) override{};

   void mobility( std::vector< double >& mobility_e, std::vector< double >& mobility_n ) const override
   {
      std::fill( mobility_e.begin(), mobility_e.end(), 1 );
      std::fill( mobility_n.begin(), mobility_n.end(), 1 );
   }
};

void CahnHilliardDGAssembly::set_evaluate_params_on_cell( std::shared_ptr< EvaluateParametersOnCell > eval) {
   evaluate_params_on_cell_ = std::move(eval);
}

void CahnHilliardDGAssembly::set_evaluate_params_on_face( std::shared_ptr< EvaluateParametersOnFace > eval ) {
   evaluate_params_on_face_ = std::move(eval);
}

CahnHilliardDGAssembly::CahnHilliardDGAssembly( lm::TransientLinearImplicitSystem& sys )
: dt_( 1e-2 )
, C_psi_( 0.05 )
, epsilon_( 0.005 )
, evaluate_params_on_cell_( std::make_shared< DefaultEvaluateParametersOnCell > ( sys ) )
, evaluate_params_on_face_( std::make_shared< DefaultEvaluateParametersOnFace > ( sys ) )
, c_evaluator( sys, "c" )
, eps_( -1 )
, penalty_c_( 1e+1 )
, penalty_mu_( 1e+1 )
, beta_( 1. / ( sys.get_mesh().mesh_dimension() - 1 ) )
, sys_( sys )
{}

void CahnHilliardDGAssembly::set_dt( double dt )
{
   dt_ = dt;
}

void CahnHilliardDGAssembly::set_C_psi( double C_psi )
{
   C_psi_ = C_psi;
}

void CahnHilliardDGAssembly::set_epsilon( double epsilon )
{
   epsilon_ = epsilon;
}

void CahnHilliardDGAssembly::set_scheme( DGScheme scheme )
{
   switch ( scheme )
   {
   case DGScheme::SIP:
      eps_ = -1;
      break;
   case DGScheme::NIP:
      eps_ = +1;
      break;
   case DGScheme::IIP:
      eps_ = 0;
      break;
   default:
      throw std::runtime_error( "unknown scheme" );
   }
}

void CahnHilliardDGAssembly::set_penalties( double penalty_c, double penalty_mu )
{
   penalty_c_ = penalty_c;
   penalty_mu_ = penalty_mu;
}

bool CahnHilliardDGAssembly::use_dg() const
{
   return sys_.variable_type( sys_.variable_number( "c" ) ).family != libMesh::LAGRANGE;
}

void CahnHilliardDGAssembly::assemble()
{
   chdg::TimeMeasurement tm( "assembly" );
   assemble_elements();
   if ( use_dg() )
      assemble_faces();
}

void CahnHilliardDGAssembly::assemble_elements()
{
   chdg::TimeMeasurement tm( "elements" );

   // this is copied directly! try to factor it into one class!!!
   const auto& mesh = sys_.get_mesh();

   const auto dim = mesh.mesh_dimension();

   const auto c_var = sys_.variable_number( "c" );
   const auto mu_var = sys_.variable_number( "mu" );

   lm::FEType fe_c_type = sys_.variable_type( c_var );
   lm::FEType fe_mu_type = sys_.variable_type( mu_var );

   if ( fe_c_type != fe_mu_type )
      throw std::runtime_error( "different finite element types not supported" );

   const auto& dof_map = sys_.get_dof_map();

   CellAssemblyData d( dim, fe_c_type, fe_c_type );
   CahnHilliardLocalSystemData ls( c_var, mu_var );

   // the values of c_old at the local quadrature points
   std::vector< lm::Real > c_old;
   // the values of the mobility at the local quadrature points
   std::vector< lm::Real > mobility;
   // the values of the rhs at the local quadrature points
   std::vector< lm::Real > rhs_c;
   std::vector< lm::Real > rhs_mu;

   // Looping through elements
   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      d.init( elem );
      ls.init( dof_map, *elem );
      evaluate_params_on_cell_->init( d );

      c_evaluator.evaluate_old( d, c_old );
      mobility.resize( d.num_quad_points() );
      rhs_c.resize( d.num_quad_points() );
      rhs_mu.resize( d.num_quad_points() );
      evaluate_params_on_cell_->mobility(mobility);
      evaluate_params_on_cell_->rhs_c(rhs_c);
      evaluate_params_on_cell_->rhs_mu(rhs_mu);

      for ( unsigned int qp = 0; qp < d.num_quad_points(); qp++ )
      {
         // const lm::Real compute_rhs_c = dt_ * c_old[qp] * ( 1 - c_old[qp] ) + c_old[qp];
         const lm::Real compute_rhs_c = c_old[qp] + dt_ * rhs_c[qp];
         const lm::Real compute_rhs_mu = c_old[qp] * C_psi_ * ( 4 * std::pow( c_old[qp], 2 ) - 6 * c_old[qp] - 1 ) + rhs_mu[qp];

         for ( unsigned int i = 0; i < d.phi.size(); i++ )
         {
            // concentration
            ls.F_c( i ) += d.JxW[qp] * compute_rhs_c * d.phi[i][qp];

            // potential
            ls.F_mu( i ) += d.JxW[qp] * compute_rhs_mu * d.phi[i][qp];

            for ( unsigned int j = 0; j < d.phi.size(); j++ )
            {
               // concentration
               ls.K_cc( i, j ) += d.JxW[qp] * d.phi[j][qp] * d.phi[i][qp];

               // concentration-potential
               ls.K_cmu( i, j ) += d.JxW[qp] * dt_ * mobility[qp] * d.dphi[j][qp] * d.dphi[i][qp];

               // potential
               ls.K_mumu( i, j ) += d.JxW[qp] * d.phi[j][qp] * d.phi[i][qp];

               // potential-concentration
               ls.K_muc( i, j ) += -d.JxW[qp] * 3.0 * C_psi_ * d.phi[j][qp] * d.phi[i][qp];
               ls.K_muc( i, j ) += -d.JxW[qp] * pow( epsilon_, 2 ) * d.dphi[j][qp] * d.dphi[i][qp];
            }
         }
      } // loop over quadrature points

      dof_map.heterogenously_constrain_element_matrix_and_vector( ls.K, ls.F, ls.dof_indices );
      sys_.matrix->add_matrix( ls.K, ls.dof_indices );
      sys_.rhs->add_vector( ls.F, ls.dof_indices );
   }
}

void CahnHilliardDGAssembly::assemble_faces()
{
   chdg::TimeMeasurement tm( "faces" );

   const auto& mesh = sys_.get_mesh();

   const auto dim = mesh.mesh_dimension();

   const auto c_var = sys_.variable_number( "c" );
   const auto mu_var = sys_.variable_number( "mu" );

   lm::FEType fe_c_type = sys_.variable_type( c_var );
   lm::FEType fe_mu_type = sys_.variable_type( mu_var );

   if ( fe_c_type != fe_mu_type )
      throw std::runtime_error( "different finite element types not supported" );

   const auto& dof_map = sys_.get_dof_map();

   chdg::LocalInteriorFaceSystem lifs;
   FaceAssemblyData face_assembly_data( dim, fe_c_type, fe_mu_type );

   std::vector< lm::Real > c_old_e;
   std::vector< lm::Real > c_old_n;

   std::vector< lm::Real > mobility_e;
   std::vector< lm::Real > mobility_n;

   std::vector< lm::Real > eps_e;
   std::vector< lm::Real > eps_n;

   for ( const auto& elem : mesh.active_local_element_ptr_range() )
   {
      for ( auto side : elem->side_index_range() )
      {
         face_assembly_data.init( elem, side );

         const lm::Elem* neighbor = elem->neighbor_ptr( side );

         // ignore exterior edges (for now :P)
         if ( face_assembly_data.is_exterior() )
            continue;

         const unsigned int elem_id = elem->id();
         const unsigned int neighbor_id = neighbor->id();

         if ( ( neighbor->active() && ( neighbor->level() == elem->level() ) && ( elem_id < neighbor_id ) ) ||
              ( neighbor->level() < elem->level() ) )
         {
            std::unique_ptr< const lm::Elem > elem_side( elem->build_side_ptr( side ) );

            // mobility
            evaluate_params_on_face_->init( face_assembly_data );
            mobility_e.resize(face_assembly_data.num_quad_points());
            mobility_n.resize(face_assembly_data.num_quad_points());
            evaluate_params_on_face_->mobility(mobility_e, mobility_n);

            // epsilon in front of laplace operator
            eps_e.resize( face_assembly_data.num_quad_points() );
            eps_n.resize( face_assembly_data.num_quad_points() );
            std::fill( eps_e.begin(), eps_e.end(), pow( epsilon_, 2 ) );
            std::fill( eps_n.begin(), eps_n.end(), pow( epsilon_, 2 ) );

            // concentration-potential laplace
            lifs.init( dof_map, mu_var, c_var, *elem, *neighbor );
            forms::diffusion::assemble_face( lifs, face_assembly_data, mobility_e, mobility_n, +dt_, eps_, penalty_c_, beta_ );
            sys_.matrix->add_matrix( lifs.K, lifs.dof_indices_psi, lifs.dof_indices_phi );

            // potential-concentration laplace
            lifs.init( dof_map, c_var, mu_var, *elem, *neighbor );
            forms::diffusion::assemble_face( lifs, face_assembly_data, eps_e, eps_n, -dt_, eps_, penalty_mu_, beta_ );
            sys_.matrix->add_matrix( lifs.K, lifs.dof_indices_psi, lifs.dof_indices_phi );
         }
      }
   }
}

} // namespace chdg