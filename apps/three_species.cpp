#include <iostream>
#include <random>
#include <utility>

#include "chdg/assembly_data.h"
#include "chdg/cahn_hilliard.h"
#include "chdg/forms.h"
#include "chdg/local_system.h"
#include "chdg/time_measurement.h"
#include "libmesh/dof_map.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/getpot.h"
#include "libmesh/libmesh.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/parameters.h"
#include "libmesh/point.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/transient_system.h"
#include "libmesh/vtk_io.h"

namespace lm = libMesh;

namespace chdg {

class SettingConfig
{
 public:
   std::string filename;
   std::string initial_solution;
   std::size_t size_x;
   std::size_t size_y;
   std::size_t size_z;
   bool use_3D;
   bool use_dg;
   lm::Order order;

 public:
   SettingConfig();

   static SettingConfig from_parameter_file( const std::string& filename );

   friend ostream& operator<<( ostream&, const SettingConfig& );
};

class CahnHilliardConfig
{
 public:
   double dt;
   double C_psi;
   double epsilon;
   double mobility_constant;

 public:
   CahnHilliardConfig();

   static CahnHilliardConfig from_parameter_file( const std::string& filename );
};

CahnHilliardConfig::CahnHilliardConfig()
: dt( 1e-6 )
, C_psi( 0 )
, epsilon( 0 )
, mobility_constant( 0 )
{}

CahnHilliardConfig CahnHilliardConfig::from_parameter_file( const std::string& filename )
{
   CahnHilliardConfig config;

   GetPot input( filename );

   config.dt = input( "dt", config.dt );
   config.C_psi = input( "C_psi", config.C_psi );
   config.epsilon = input( "epsilon", config.epsilon );
   config.mobility_constant = input( "mobility_constant", config.mobility_constant );

   return config;
}

SettingConfig::SettingConfig()
: initial_solution( "random" )
, size_x( 2 << 5 )
, size_y( 2 << 5 )
, size_z( 2 << 5 )
, use_3D( false )
, use_dg( true )
, order( static_cast< lm::Order >( 1 ) )
{}

SettingConfig SettingConfig::from_parameter_file( const std::string& filename )
{
   SettingConfig config;

   GetPot input( filename );

   config.filename = filename;
   config.initial_solution = input( "initial_solution", "undefined" );
   config.use_3D = input( "use3D", false );

   config.size_x = 1u << input( "size_x", 6u );
   config.size_y = 1u << input( "size_y", 6u );
   config.size_z = 1u << input( "size_z", 6u );

   config.use_dg = input( "use_dg", true );
   config.order = static_cast< lm::Order >( input( "order", 1 ) );

   return config;
}

ostream& operator<<( ostream& os, const SettingConfig& config )
{
   os << "filename " << config.filename;
   os << "initial solution " << config.initial_solution;
   os << "size_x " << config.size_x;
   os << "size_y " << config.size_y;
   os << "size_z " << config.size_z;
   os << "use_3D " << config.use_3D;
   return os;
}

/*! @brief Random initial condition for the concentration. */
lm::Number initial_condition_cahnhilliard_random( const lm::Point& p,
                                                  const lm::Parameters& es,
                                                  const std::string& system_name,
                                                  const std::string& var_name );

/*! @brief Smooth circular initial condition for the concentration. */
lm::Number initial_condition_cahnhilliard_circle_2d( const lm::Point& p,
                                                     const lm::Parameters& es,
                                                     const std::string& system_name,
                                                     const std::string& var_name );

lm::Number initial_condition_cahnhilliard_circle_3d( const lm::Point& p,
                                                     const lm::Parameters& es,
                                                     const std::string& system_name,
                                                     const std::string& var_name );

lm::Number initial_condition_cahnhilliard_random( const lm::Point& p,
                                                  const lm::Parameters& es,
                                                  const std::string& system_name,
                                                  const std::string& var_name )
{
   libmesh_assert_equal_to( system_name, "CahnHilliard" );

   static std::default_random_engine generator( 2 + es.get< unsigned int >( "rank" ) );
   const auto mean = 0.5;
   const auto interval = 0.3;
   std::uniform_real_distribution< double > distribution( mean - interval, mean + interval );

   if ( var_name == "c" )
   {
      return distribution( generator );
   }

   return 0;
}

lm::Number initial_condition_cahnhilliard_circle_3d( const lm::Point& p,
                                                     const lm::Parameters& es,
                                                     const std::string& system_name,
                                                     const std::string& var_name )
{
   // circle with middle point (1/2,1/2) and radius a
   const auto a = 0.25;
   const auto dist_squared = std::pow( p( 0 ) - 0.5, 2 ) + std::pow( p( 1 ) - 0.5, 2 ) + std::pow( p( 2 ) - 0.5, 2 );

   if ( dist_squared < a * a )
      return std::exp( 1. - 1. / ( 1. - dist_squared / a * a ) );

   return 0;
}

lm::Number initial_condition_cahnhilliard_circle_2d( const lm::Point& p,
                                                     const lm::Parameters& es,
                                                     const std::string& system_name,
                                                     const std::string& var_name )
{
   // circle with middle point (1/2,1/2) and radius a
   const auto a = 0.25;
   const auto dist_squared = std::pow( p( 0 ) - 0.5, 2 ) + std::pow( p( 1 ) - 0.5, 2 );

   if ( dist_squared < a * a )
      return std::exp( 1. - 1. / ( 1. - dist_squared / a * a ) );

   return 0;
}

class SpeciesConfig
{
 public:
   double lambda_pro_h;
   double lambda_deg_h;

   double lambda_pro_p;
   double lambda_deg_p;

   double lambda_ph;
   double lambda_hp;
   double lambda_hn;

   double sigma_ph;
   double sigma_hp;

   double mobility;
};

class HypoxicEvalParamsOnCellFct
{
 public:
   HypoxicEvalParamsOnCellFct( const lm::TransientLinearImplicitSystem& hypoxic,
                               const lm::TransientLinearImplicitSystem& prolific,
                               const lm::TransientLinearImplicitSystem& necrotic,
                               const SpeciesConfig& config )
   : hypoxic_evaluator( hypoxic, "hypoxic" )
   , prolific_evaluator( prolific, "prolific" )
   , necrotic_evaluator( necrotic, "necrotic" )
   , config( config )
   {}

   void operator()( const CellAssemblyData& data, std::vector< double >& mobility, std::vector< double >& rhs ) const
   {
      hypoxic_evaluator.evaluate_old( data, hyp );
      prolific_evaluator.evaluate_old( data, pro );
      necrotic_evaluator.evaluate_old( data, nec );
      evaluate_nutrients_old( data, nut );

      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const lm::Real tumor = hyp[qp] + pro[qp] + nec[qp];

         // mobility
         mobility[qp] = config.mobility * pow( util::cutoff( hyp[qp] ) * util::cutoff( 1. - tumor ), 2 );

         // rhs
         rhs[qp] = 0;
         rhs[qp] += config.lambda_pro_h * nut[qp] * hyp[qp] * ( 1 - tumor );
         rhs[qp] -= config.lambda_deg_h * hyp[qp];
         rhs[qp] += config.lambda_ph * util::heavyside( config.sigma_ph - nut[qp] ) * pro[qp];
         rhs[qp] -= config.lambda_hp * util::heavyside( nut[qp] - config.sigma_hp ) * hyp[qp];
         rhs[qp] -= config.lambda_hn * util::heavyside( config.sigma_hp - nut[qp] ) * hyp[qp];
      }
   }

 private:
   EvaluatorOnQuadraturePoints hypoxic_evaluator;
   EvaluatorOnQuadraturePoints prolific_evaluator;
   EvaluatorOnQuadraturePoints necrotic_evaluator;

   SpeciesConfig config;

   mutable std::vector< lm::Real > hyp;
   mutable std::vector< lm::Real > pro;
   mutable std::vector< lm::Real > nec;
   mutable std::vector< lm::Real > nut;

   static void evaluate_nutrients_old( const CellAssemblyData& data, std::vector< double >& values )
   {
      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const auto& p = data.xyz[qp];
         values[qp] = ( p( 0 ) + p( 1 ) ) / 2;
      }
   }
};

class HypoxicEvalParamsOnFaceFct
{
 public:
   HypoxicEvalParamsOnFaceFct( const lm::TransientLinearImplicitSystem& hypoxic,
                               const lm::TransientLinearImplicitSystem& prolific,
                               const lm::TransientLinearImplicitSystem& necrotic,
                               const SpeciesConfig& config )
   : hypoxic_evaluator( hypoxic, "hypoxic" )
   , prolific_evaluator( prolific, "prolific" )
   , necrotic_evaluator( necrotic, "necrotic" )
   , config( config )
   {}

   void operator()( const FaceAssemblyData& data, std::vector< double >& mobility_e, std::vector< double >& mobility_n ) const
   {
      hypoxic_evaluator.evaluate_old( data, hyp_e, hyp_n );
      prolific_evaluator.evaluate_old( data, pro_e, pro_n );
      necrotic_evaluator.evaluate_old( data, nec_e, nec_n );

      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const auto tumor_e = hyp_e[qp] + pro_e[qp] + nec_e[qp];
         const auto tumor_n = hyp_n[qp] + pro_n[qp] + nec_n[qp];

         mobility_e[qp] = config.mobility * pow( util::cutoff( hyp_e[qp] ) * util::cutoff( 1. - tumor_e ), 2 );
         mobility_n[qp] = config.mobility * pow( util::cutoff( hyp_n[qp] ) * util::cutoff( 1. - tumor_n ), 2 );
      }
   }

 private:
   EvaluatorOnQuadraturePoints hypoxic_evaluator;
   EvaluatorOnQuadraturePoints prolific_evaluator;
   EvaluatorOnQuadraturePoints necrotic_evaluator;

   SpeciesConfig config;

   mutable std::vector< lm::Real > hyp_e;
   mutable std::vector< lm::Real > pro_e;
   mutable std::vector< lm::Real > nec_e;
   mutable std::vector< lm::Real > hyp_n;
   mutable std::vector< lm::Real > pro_n;
   mutable std::vector< lm::Real > nec_n;
};

class ProlificEvalParamsOnCellFct
{
 public:
   ProlificEvalParamsOnCellFct( const lm::TransientLinearImplicitSystem& hypoxic,
                                const lm::TransientLinearImplicitSystem& prolific,
                                const lm::TransientLinearImplicitSystem& necrotic,
                                const SpeciesConfig& config )
   : hypoxic_evaluator( hypoxic, "hypoxic" )
   , prolific_evaluator( prolific, "prolific" )
   , necrotic_evaluator( necrotic, "necrotic" )
   , config( config )
   {}

   void operator()( const CellAssemblyData& data, std::vector< double >& mobility, std::vector< double >& rhs ) const
   {
      hypoxic_evaluator.evaluate_old( data, hyp );
      prolific_evaluator.evaluate_old( data, pro );
      necrotic_evaluator.evaluate_old( data, nec );
      evaluate_nutrients_old( data, nut );

      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const lm::Real tumor = hyp[qp] + pro[qp] + nec[qp];

         // mobility
         mobility[qp] = config.mobility * pow( util::cutoff( pro[qp] ) * util::cutoff( 1. - tumor ), 2 );

         // rhs
         rhs[qp] = 0;
         rhs[qp] += config.lambda_pro_p * nut[qp] * pro[qp] * ( 1 - tumor );
         rhs[qp] -= config.lambda_deg_p * pro[qp];
         rhs[qp] -= config.lambda_ph * util::heavyside( config.lambda_ph - nut[qp] ) * pro[qp];
         rhs[qp] += config.lambda_hp * util::heavyside( nut[qp] - config.sigma_hp ) * hyp[qp];
      }
   }

 private:
   EvaluatorOnQuadraturePoints hypoxic_evaluator;
   EvaluatorOnQuadraturePoints prolific_evaluator;
   EvaluatorOnQuadraturePoints necrotic_evaluator;

   SpeciesConfig config;

   mutable std::vector< lm::Real > hyp;
   mutable std::vector< lm::Real > pro;
   mutable std::vector< lm::Real > nec;
   mutable std::vector< lm::Real > nut;

   static void evaluate_nutrients_old( const CellAssemblyData& data, std::vector< double >& values )
   {
      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const auto& p = data.xyz[qp];
         values[qp] = ( p( 0 ) + p( 1 ) ) / 2;
      }
   }
};

class ProlificEvalParamsOnFaceFct
{
 public:
   ProlificEvalParamsOnFaceFct( const lm::TransientLinearImplicitSystem& hypoxic,
                                const lm::TransientLinearImplicitSystem& prolific,
                                const lm::TransientLinearImplicitSystem& necrotic,
                                const SpeciesConfig& config )
   : hypoxic_evaluator( hypoxic, "hypoxic" )
   , prolific_evaluator( prolific, "prolific" )
   , necrotic_evaluator( necrotic, "necrotic" )
   , config( config )
   {}

   void operator()( const FaceAssemblyData& data, std::vector< double >& mobility_e, std::vector< double >& mobility_n ) const
   {
      hypoxic_evaluator.evaluate_old( data, hyp_e, hyp_n );
      prolific_evaluator.evaluate_old( data, pro_e, pro_n );
      necrotic_evaluator.evaluate_old( data, nec_e, nec_n );

      for ( unsigned int qp = 0; qp < data.num_quad_points(); qp += 1 )
      {
         const auto tumor_e = hyp_e[qp] + pro_e[qp] + nec_e[qp];
         const auto tumor_n = hyp_n[qp] + pro_n[qp] + nec_n[qp];

         mobility_e[qp] = config.mobility * pow( util::cutoff( pro_e[qp] ) * util::cutoff( 1. - tumor_e ), 2 );
         mobility_n[qp] = config.mobility * pow( util::cutoff( pro_n[qp] ) * util::cutoff( 1. - tumor_n ), 2 );
      }
   }

 private:
   EvaluatorOnQuadraturePoints hypoxic_evaluator;
   EvaluatorOnQuadraturePoints prolific_evaluator;
   EvaluatorOnQuadraturePoints necrotic_evaluator;

   SpeciesConfig config;

   mutable std::vector< lm::Real > hyp_e;
   mutable std::vector< lm::Real > pro_e;
   mutable std::vector< lm::Real > nec_e;
   mutable std::vector< lm::Real > hyp_n;
   mutable std::vector< lm::Real > pro_n;
   mutable std::vector< lm::Real > nec_n;
};

class CahnHilliardTimestepper
{
 public:
   /*! @brief Constructor */
   CahnHilliardTimestepper( const std::string& system_name, const std::string& filename, lm::EquationSystems& sys );

   /*! @brief Run model. */
   void run();

 private:
   void write_system( const unsigned int& t_step );

 private:
   /*! @brief The equation system. */
   lm::EquationSystems& sys_;

   /*! @brief The equation system. */
   lm::TransientLinearImplicitSystem& hyp_;
   lm::TransientLinearImplicitSystem& pro_;
   lm::TransientLinearImplicitSystem& nec_;

   CahnHilliardConfig cahn_hilliard_config;

   /*! @brief Assembly objects. */
   CahnHilliardDGAssembly cahn_hilliard_assembly;

   /*! @brief Current time step. */
   std::size_t step_;

   /*! @brief Total number of time steps. */
   std::size_t steps_;

   /*! @brief Current time. */
   double time_;

   /*! @brief Current time step size. */
   double dt_;
};

// Model class
CahnHilliardTimestepper::CahnHilliardTimestepper( const std::string& system_name,
                                                  const std::string& filename,
                                                  lm::EquationSystems& sys )
: sys_( sys )
, ch_( sys_.get_system< lm::TransientLinearImplicitSystem >( "CahnHilliard" ) )
, cahn_hilliard_config( CahnHilliardConfig::from_parameter_file( filename ) )
, cahn_hilliard_assembly( ch_ )
, steps_( 100 )
, step_( 0 )
, time_( 0 )
, dt_( 1e-2 )
{
   ch_.attach_assemble_object( cahn_hilliard_assembly );
   sys.init();

   cahn_hilliard_assembly.set_C_psi( cahn_hilliard_config.C_psi );
   cahn_hilliard_assembly.set_epsilon( cahn_hilliard_config.epsilon );
   cahn_hilliard_assembly.set_dt( cahn_hilliard_config.dt );
   cahn_hilliard_assembly.set_evaluate_params_on_cell( EvalParamsOnCellFct( ch_, "c", cahn_hilliard_config.mobility_constant ) );
   cahn_hilliard_assembly.set_evaluate_params_on_face( EvalParamsOnFaceFct( ch_, "c", cahn_hilliard_config.mobility_constant ) );

   GetPot input( filename );
   dt_ = input( "dt", 1e-2 );
   const auto final_time = input( "final_time", 100 * dt_ );
   steps_ = std::ceil( final_time / dt_ );
}

void CahnHilliardTimestepper::run()
{
   write_system( 0 );

   do
   {
      // Prepare time step
      step_++;
      time_ += dt_;

      std::cout << "Time step: " << std::to_string( step_ ) << ", time: " << std::to_string( time_ ) << std::endl;
      std::cout << ch_.current_local_solution->max() << " " << ch_.current_local_solution->min() << std::endl;

      // Prepare solution for the current timestep
      *( ch_.old_local_solution ) = *( ch_.current_local_solution );

      // solve cahn-hilliard system, where the old solution is our initial guess
      {
         chdg::TimeMeasurement tm( "solve" );
         ch_.solve();
      }

      // write solution to disk
      write_system( step_ );

   } while ( step_ < steps_ );
}

void CahnHilliardTimestepper::write_system( const unsigned int& t_step )
{
   lm::VTKIO( sys_.get_mesh() ).write_equation_systems( "output/cahnhilliard_" + std::to_string( t_step ) + ".pvtu", sys_ );
   lm::ExodusII_IO( sys_.get_mesh() ).write_equation_systems( "output/cahnhilliard_" + std::to_string( t_step ) + ".e", sys_ );
   lm::ExodusII_IO( sys_.get_mesh() )
       .write_discontinuous_exodusII( "output/cahnhilliard_" + std::to_string( t_step ) + ".e", sys_ );
}

} // namespace chdg

void initial_condition( lm::EquationSystems& es, const std::string& system_name )
{
   auto& sys = es.get_system< lm::TransientLinearImplicitSystem >( system_name );

   const auto& config = es.parameters.get< chdg::SettingConfig >( "setting_config" );

   if ( system_name == "CahnHilliard" )
   {
      if ( config.initial_solution == "random" )
      {
         sys.project_solution( chdg::initial_condition_cahnhilliard_random, nullptr, es.parameters );
      }
      else if ( config.initial_solution == "circle" )
      {
         if ( config.use_3D )
         {
            sys.project_solution( chdg::initial_condition_cahnhilliard_circle_3d, nullptr, es.parameters );
         }
         else
         {
            sys.project_solution( chdg::initial_condition_cahnhilliard_circle_2d, nullptr, es.parameters );
         }
      }
      else
      {
         throw std::runtime_error( "unknown initial solution type " + config.initial_solution );
      }
   }
   else
   {
      return;
   }
}

int main( int argc, char** argv )
{
   lm::LibMeshInit init( argc, argv );

   std::string config_filename{ "cahn_hilliard_ex.in" };

   //Parse the input file
   const auto config = chdg::SettingConfig::from_parameter_file( config_filename );

   const auto& comm = init.comm();

   lm::Mesh mesh( comm );
   if ( config.use_3D )
   {
      lm::MeshTools::Generation::build_cube( mesh, config.size_x, config.size_y, config.size_z, 0, 1, 0, 1, 0, 1, lm::HEX8 );
   }
   else
   {
      //lm::MeshTools::Generation::build_square( mesh, config.size_x, config.size_y, 0., 1, 0., 1, lm::QUAD4 );
      lm::MeshTools::Generation::build_square( mesh, config.size_x, config.size_y, 0., 1, 0., 1, lm::TRI3 );
   }

   auto sys = std::make_shared< lm::EquationSystems >( mesh );
   sys->parameters.set< chdg::SettingConfig >( "setting_config" ) = config;
   sys->parameters.set< unsigned int >( "rank" ) = sys->comm().rank();

   // Add systems, variables and assemble
   auto& hyp = sys->add_system< lm::TransientLinearImplicitSystem >( "Hypoxic" );
   auto& pro = sys->add_system< lm::TransientLinearImplicitSystem >( "Prolific" );
   auto& nec = sys->add_system< lm::TransientLinearImplicitSystem >( "Necrotic" );

   if ( config.use_dg )
   {
      // ch.add_variable( "c", config.order, lm::MONOMIAL );
      // ch.add_variable( "mu", config.order, lm::MONOMIAL );
      ch.add_variable( "c", config.order, lm::L2_LAGRANGE );
      ch.add_variable( "mu", config.order, lm::L2_LAGRANGE );
   }
   else
   {
      ch.add_variable( "c", config.order );
      ch.add_variable( "mu", config.order );
   }

   ch.attach_init_function( initial_condition );

   chdg::CahnHilliardTimestepper timestepper( "CahnHilliard", config_filename, *sys );
   timestepper.run();

   return 0;
}