#define TEST_UTILS_PROVIDE_MAIN
#include "chdg/integrator.h"
#include "chdg/mark_boundary.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exact_solution.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "test_utils.h"

using namespace libMesh;

double solve( unsigned int order, unsigned int dim, double eps, double penalty, ExactSolution::ValueFunctionPointer solution );

Number exact_solution_p1_2d( const Point& p, const Parameters&, const std::string&, const std::string& );
Number exact_solution_p2_2d( const Point& p, const Parameters&, const std::string&, const std::string& );
Number exact_solution_p1_3d( const Point& p, const Parameters&, const std::string&, const std::string& );
Number exact_solution_p2_3d( const Point& p, const Parameters&, const std::string&, const std::string& );

TEST_CASE( "the elliptic equation yields exact solutions", "[elliptic_equation]" )
{
   SECTION( "NIPG with P1 in 2D" )
   {
      const auto error = solve( 1, 2, +1, 1e-1, exact_solution_p1_2d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "NIPG with P2 in 2D" )
   {
      const auto error = solve( 2, 2, +1, 1e-1, exact_solution_p2_2d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "NIPG with P1 in 3D" )
   {
      const auto error = solve( 1, 3, +1, 1e-1, exact_solution_p1_3d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "NIPG with P2 in 3D" )
   {
      const auto error = solve( 2, 3, +1, 1e-1, exact_solution_p2_3d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "SIPG with P1 in 2D" )
   {
      const auto error = solve( 1, 2, -1, 1e+1, exact_solution_p1_2d );
      REQUIRE( error < 1e-14 );
   }

   SECTION( "SIPG with P2 in 2D" )
   {
      const auto error = solve( 2, 2, -1, 1e+1, exact_solution_p2_2d );
      std::cout << error << std::endl;
      REQUIRE( error < 1e-13 );
   }

   SECTION( "SIPG with P1 in 3D" )
   {
      const auto error = solve( 1, 3, -1, 1e+1, exact_solution_p1_3d );
      REQUIRE( error < 1e-13 );
   }

   SECTION( "SIPG with P2 in 3D" )
   {
      const auto error = solve( 2, 3, -1, 1e+1, exact_solution_p2_3d );
      REQUIRE( error < 1e-13 );
   }
}

Number exact_solution_p1_2d( const Point& p, const Parameters&, const std::string&, const std::string& )
{
   const Real x = p( 0 );
   const Real y = p( 1 );

   return 2 * x + y;
}

Number exact_solution_p2_2d( const Point& p, const Parameters&, const std::string&, const std::string& )
{
   const Real x = p( 0 );
   const Real y = p( 1 );

   return 2 * x * y + y - 1;
}

Number exact_solution_p1_3d( const Point& p, const Parameters&, const std::string&, const std::string& )
{
   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return 2 * x + y - z;
}

Number exact_solution_p2_3d( const Point& p, const Parameters&, const std::string&, const std::string& )
{
   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return 2 * x * y + y - 1 - z * x + 0.5 * x;
}

double solve( unsigned int order, unsigned int dim, double eps, double penalty, ExactSolution::ValueFunctionPointer solution )
{
   const uint uniform_refinement_steps = 2;

   Mesh mesh( test_utils::lmInit->comm() );

   if ( dim == 1 )
      MeshTools::Generation::build_line( mesh, 1, -1., 0. );
   else if ( dim == 2 )
      MeshTools::Generation::build_square( mesh, 4, 4, -1., 1., -1., 1., TRI3 );
   else
      MeshTools::Generation::build_cube( mesh, 4, 4, 4, -1., 1., -1., 1., -1., 1., HEX8 );

   // mark the whole boundary as dirichlet
   mesh.get_boundary_info().clear();
   chdg::mark_boundary( mesh, 0, []( const Point& ) { return true; } );

   MeshRefinement mesh_refinement( mesh );
   for ( unsigned int rstep = 0; rstep < uniform_refinement_steps; rstep++ )
      mesh_refinement.uniformly_refine( 1 );

   EquationSystems equation_system( mesh );

   equation_system.parameters.set< Real >( "linear solver tolerance" ) = 1e-16;
   equation_system.parameters.set< unsigned int >( "linear solver maximum iterations" ) = 1000;
   equation_system.parameters.set< Real >( "penalty" ) = penalty;

   auto& elliptic_system = equation_system.add_system< LinearImplicitSystem >( "EllipticDG" );

   const auto u_var_id = elliptic_system.add_variable( "u", static_cast< const Order >( order ), MONOMIAL );
   // elliptic_system.add_variable( "u", static_cast< const Order >(p_order), XYZ );

   chdg::integrator::EllipticIntegrator elliptic_integrator( elliptic_system );
   elliptic_integrator.set_trial_function( u_var_id );
   elliptic_integrator.set_test_function( u_var_id );
   elliptic_integrator.set_tau( 2. );
   elliptic_integrator.set_dirichlet_boundary_id( 0 );
   elliptic_integrator.set_dirichlet_boundary_value( [&]( const auto& points, auto& values ) {
      for ( std::size_t qp = 0; qp < points.size(); qp += 1 )
         values[qp] = solution( points[qp], equation_system.parameters, "", "" );
   } );
   elliptic_integrator.set_eps( eps );
   elliptic_integrator.set_penalty( penalty );
   elliptic_system.attach_assemble_object( elliptic_integrator );

   equation_system.init();

   ExactSolution exact_sol( equation_system );
   exact_sol.attach_exact_value( solution );

   elliptic_system.solve();

   libMesh::out << "System has: " << equation_system.n_active_dofs() << " degrees of freedom." << std::endl;

   libMesh::out << "Linear solver converged at step: " << elliptic_system.n_linear_iterations()
                << ", final residual: " << elliptic_system.final_linear_residual() << std::endl;

   // Compute the error
   exact_sol.compute_error( "EllipticDG", "u" );

   return exact_sol.l2_error( "EllipticDG", "u" );
}
