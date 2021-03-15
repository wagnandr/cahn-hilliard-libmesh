#define TEST_UTILS_PROVIDE_MAIN
#include "chdg/integrator.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exact_solution.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/transient_system.h"
#include "test_utils.h"

using namespace libMesh;

using VelocityField = std::function< Point( const Point& ) >;

double solve( unsigned int order, unsigned int dim, VelocityField velocity_field, ExactSolution::ValueFunctionPointer solution );

Number exact_solution_p1_2d( const Point& p, const Parameters&, const std::string&, const std::string& );
Number exact_solution_p1_3d( const Point& p, const Parameters&, const std::string&, const std::string& );
Number exact_solution_p2_2d( const Point& p, const Parameters& parameters, const std::string&, const std::string& );
Number exact_solution_p2_3d( const Point& p, const Parameters&, const std::string&, const std::string& );

Point velocity_2d_top_right( const Point& p );
Point velocity_2d_up( const Point& p );
Point velocity_3d_top_right( const Point& p );
Point velocity_3d_up( const Point& p );

TEST_CASE( "the advection equation yields exact solutions", "[advection_equation]" )
{
   SECTION( "P1 in 2D" )
   {
      const auto error = solve( 1, 2, velocity_2d_top_right, exact_solution_p1_2d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "P2 in 2D" )
   {
      const auto error = solve( 2, 2, velocity_2d_up, exact_solution_p2_2d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "P1 in 3D" )
   {
      const auto error = solve( 1, 3, velocity_3d_top_right, exact_solution_p1_3d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "P2 in 3D" )
   {
      const auto error = solve( 2, 3, velocity_3d_up, exact_solution_p2_3d );
      REQUIRE( error < 1e-14 );
   }
}

/// Analytic solution for a normalized top right velocity in 2D.
Number exact_solution_p1_2d( const Point& p, const Parameters& parameters, const std::string&, const std::string& )
{
   const double t = parameters.get< double >( "t" );

   const Real x = p( 0 );
   const Real y = p( 1 );

   return 2.5 / std::sqrt( 2 ) * t - 2 * y - 0.5 * x;
}

/// Analytic solution for a normalized up velocity in 2D.
Number exact_solution_p2_2d( const Point& p, const Parameters& parameters, const std::string&, const std::string& )
{
   const double t = parameters.get< double >( "t" );

   const Real x = p( 0 );
   const Real y = p( 1 );

   return x * t - x * y - 0.5 * x;
}

/// Analytic solution for a normalized up velocity in 3D.
Number exact_solution_p2_3d( const Point& p, const Parameters& parameters, const std::string&, const std::string& )
{
   const double t = parameters.get< double >( "t" );

   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return ( x - 0.5 * y ) * t - ( x - 0.5 * y ) * z - 0.5 * x;
}

/// Analytic solution for a normalized top right velocity in 2D.
Number exact_solution_p1_3d( const Point& p, const Parameters& parameters, const std::string&, const std::string& )
{
   const double t = parameters.get< double >( "t" );

   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return 1.5 / std::sqrt( 3 ) * t - 2 * y + z - 0.5 * x;
}

Point velocity_2d_top_right( const Point& p )
{
   return Point( 1. / std::sqrt( 2 ), 1. / std::sqrt( 2 ) );
}

Point velocity_2d_up( const Point& p )
{
   return Point( 0, 1 );
}

Point velocity_3d_top_right( const Point& p )
{
   return Point( 1. / std::sqrt( 3 ), 1. / std::sqrt( 3 ), 1. / std::sqrt( 3 ) );
}

Point velocity_3d_up( const Point& p )
{
   return Point( 0., 0., 1. );
}

double solve( unsigned int order, unsigned int dim, VelocityField velocity_field, ExactSolution::ValueFunctionPointer solution )
{
   const uint uniform_refinement_steps = 1;

   const double t_max = 1;
   const uint num_timesteps = 1u << 4u;
   const double dt = t_max / num_timesteps;

   // exodus output for debugging
   const bool exodus_output = true;
   std::string exodus_filename = "transient.exo";

   Mesh mesh( test_utils::lmInit->comm() );

   if ( dim == 1 )
      MeshTools::Generation::build_line( mesh, 1, -1., 0. );
   else if ( dim == 2 )
      MeshTools::Generation::build_square( mesh, 4, 4, -1., 1., -1., 1., TRI3 );
   else
      MeshTools::Generation::build_cube( mesh, 4, 4, 4, -1., 1., -1., 1., -1., 1., HEX8 );

   MeshRefinement mesh_refinement( mesh );
   for ( unsigned int rstep = 0; rstep < uniform_refinement_steps; rstep++ )
      mesh_refinement.uniformly_refine( uniform_refinement_steps );

   EquationSystems equation_system( mesh );

   equation_system.parameters.set< Real >( "linear solver tolerance" ) = 1e-16;
   equation_system.parameters.set< unsigned int >( "linear solver maximum iterations" ) = 1000;

   auto& advection_system = equation_system.add_system< TransientLinearImplicitSystem >( "AdvectionDG" );

   const auto u_var_id = advection_system.add_variable( "u", static_cast< const Order >( order ), MONOMIAL );

   chdg::integrator::TransientImplicitAdvectionIntegrator advection_scheme( advection_system );
   advection_scheme.set_trial_function( u_var_id );
   advection_scheme.set_test_function( u_var_id );
   advection_scheme.set_dt( dt );
   advection_scheme.get_advection().set_velocity_field( velocity_field );
   advection_scheme.get_advection().set_inflow_boundary_value(
       [&]( const Point& p ) { return solution( p, equation_system.parameters, "", "" ); } );

   advection_system.attach_assemble_object( advection_scheme );

   equation_system.init();

   ExactSolution exact_sol( equation_system );
   exact_sol.attach_exact_value( solution );

   // initial conditions
   advection_system.time = 0;
   equation_system.parameters.set< Real >( "t" ) = advection_system.time;
   advection_system.project_solution( solution, nullptr, equation_system.parameters );

   exact_sol.compute_error( "AdvectionDG", "u" );
   libMesh::out << "error " << exact_sol.l2_error( "AdvectionDG", "u" ) << std::endl;

   if ( exodus_output )
   {
      ExodusII_IO( mesh ).write_equation_systems( exodus_filename, equation_system );
   }

   if ( exodus_output )
   {
      ExodusII_IO exo( mesh );
      exo.append( true );
      exo.write_timestep( exodus_filename, equation_system, 1, advection_system.time );
   }

   // loop over the time steps
   for ( uint t_idx = 0; t_idx < num_timesteps; t_idx += 1 )
   {
      advection_system.time += dt;
      equation_system.parameters.set< Real >( "t" ) = advection_system.time;
      libMesh::out << "solving time step ";

      {
         std::ostringstream sout;

         sout << std::setw( 2 ) << std::right << dt << ", time=" << std::fixed << std::setw( 6 ) << std::setprecision( 3 )
              << std::setfill( '0' ) << std::left << advection_system.time << "...";

         libMesh::out << sout.str() << std::endl;
      }

      *advection_system.old_local_solution = *advection_system.current_local_solution;

      advection_system.solve();

      exact_sol.compute_error( "AdvectionDG", "u" );

      libMesh::out << "error " << exact_sol.l2_error( "AdvectionDG", "u" ) << std::endl;

      if ( exodus_output )
      {
         ExodusII_IO exo( mesh );
         exo.append( true );
         exo.write_timestep( exodus_filename, equation_system, t_idx + 2, advection_system.time );
      }
   }

   // we return the last error
   return exact_sol.l2_error( "AdvectionDG", "u" );
}
