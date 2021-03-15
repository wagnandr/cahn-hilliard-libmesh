// The libMesh Finite Element Library.
// Copyright (C) 2002-2020 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// <h1>Miscellaneous Example 5 - Interior Penalty Discontinuous Galerkin</h1>
// \author Lorenzo Botti
// \date 2010
//
// This example is based on Adaptivity Example 3, but uses an
// Interior Penalty Discontinuous Galerkin formulation.

#include <iostream>

// LibMesh include files.
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"
#include "libmesh/dense_vector.h"
#include "libmesh/discontinuity_measure.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_solver_package.h"
#include "libmesh/equation_systems.h"
#include "libmesh/error_vector.h"
#include "libmesh/exact_solution.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/fe.h"
#include "libmesh/fe_interface.h"
#include "libmesh/getpot.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/libmesh.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/transient_system.h"
//#define QORDER TWENTYSIXTH

#include "chdg/integrator.h"
#include "chdg/mark_boundary.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

Number exact_solution( const Point& p, const Parameters& parameters, const std::string&, const std::string& )
{
   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return 2 * x + y;
}

Number exact_solution2( const Point& p )
{
   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   return 2 * x + y;
}

// We now define the gradient of the exact solution, again being careful
// to obtain an angle from atan2 in the correct
// quadrant.
Gradient exact_derivative( const Point& p,
                           const Parameters& parameters, // es parameters
                           const std::string&,           // sys_name, not needed
                           const std::string& )          // unk_name, not needed
{
   // Gradient value to be returned.
   Gradient gradu;

   // x and y coordinates in space
   const Real x = p( 0 );
   const Real y = p( 1 );
   const Real z = p( 2 );

   gradu( 0 ) = 2;
   gradu( 1 ) = y;

   return gradu;

   if ( parameters.get< bool >( "singularity" ) )
   {
      libmesh_assert_not_equal_to( x, 0. );

      // For convenience...
      const Real tt = 2. / 3.;
      const Real ot = 1. / 3.;

      // The value of the radius, squared
      const Real r2 = x * x + y * y;

      // The boundary value, given by the exact solution,
      // u_exact = r^(2/3)*sin(2*theta/3).
      Real theta = atan2( y, x );

      // Make sure 0 <= theta <= 2*pi
      if ( theta < 0 )
         theta += 2. * libMesh::pi;

      // du/dx
      gradu( 0 ) = tt * x * pow( r2, -tt ) * sin( tt * theta ) -
                   pow( r2, ot ) * cos( tt * theta ) * tt / ( 1. + y * y / x / x ) * y / x / x;
      gradu( 1 ) =
          tt * y * pow( r2, -tt ) * sin( tt * theta ) + pow( r2, ot ) * cos( tt * theta ) * tt / ( 1. + y * y / x / x ) * 1. / x;
      gradu( 2 ) = 1.;
   }
   else
   {
      gradu( 0 ) = -sin( x ) * exp( y ) * ( 1. - z );
      gradu( 1 ) = cos( x ) * exp( y ) * ( 1. - z );
      gradu( 2 ) = -cos( x ) * exp( y );
   }
   return gradu;
}

void assemble_ellipticdg( EquationSystems& es, const std::string& libmesh_dbg_var( system_name ) )
{
   auto& ellipticdg_system = es.get_system< LinearImplicitSystem >( "EllipticDG" );

   chdg::integrator::EllipticIntegrator elliptic_integrator( ellipticdg_system );
   elliptic_integrator.set_trial_function( 0 );
   elliptic_integrator.set_test_function( 0 );
   elliptic_integrator.set_dirichlet_boundary_id( 0 );
   elliptic_integrator.set_dirichlet_boundary_value(
       [&]( const Point& p ) { return exact_solution( p, es.parameters, "", "" ); } );
   elliptic_integrator.set_eps( +1 );
   elliptic_integrator.set_penalty( +1e-1 );

   ellipticdg_system.attach_assemble_object( elliptic_integrator );
}

int main( int argc, char** argv )
{
   LibMeshInit init( argc, argv );

   // This example requires a linear solver package.
   libmesh_example_requires( libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                             "--enable-petsc, --enable-trilinos, or --enable-eigen" );

   // Skip adaptive examples on a non-adaptive libMesh build
#ifndef LIBMESH_ENABLE_AMR
   libmesh_example_requires( false, "--enable-amr" );
#else

   //Parse the input file
   GetPot input_file( "miscellaneous_ex5.in" );

   //Read in parameters from the input file
   const unsigned int adaptive_refinement_steps = input_file( "max_adaptive_r_steps", 3 );
   const unsigned int uniform_refinement_steps = input_file( "uniform_h_r_steps", 3 );
   const Real refine_fraction = input_file( "refine_fraction", 0.5 );
   const Real coarsen_fraction = input_file( "coarsen_fraction", 0. );
   const unsigned int max_h_level = input_file( "max_h_level", 10 );
   const std::string refinement_type = input_file( "refinement_type", "p" );
   Order p_order = static_cast< Order >( input_file( "p_order", 1 ) );
   const std::string element_type = input_file( "element_type", "tensor" );
   const Real penalty = input_file( "ip_penalty", 10. );
   const bool singularity = input_file( "singularity", true );
   const unsigned int dim = input_file( "dimension", 3 );

   // Skip higher-dimensional examples on a lower-dimensional libMesh build
   libmesh_example_requires( dim <= LIBMESH_DIM, "2D/3D support" );

   // Create a mesh, with dimension to be overridden later, distributed
   // across the default MPI communicator.
   Mesh mesh( init.comm() );

   if ( dim == 1 )
      MeshTools::Generation::build_line( mesh, 1, -1., 0. );
   else if ( dim == 2 )
      MeshTools::Generation::build_square( mesh, 4, 4, -1., 1., -1., 1., TRI3 );
   else
      MeshTools::Generation::build_cube( mesh, 4, 4, 4, -1., 1., -1., 1., -1., 1., HEX8 );

   // mark the whole boundary as dirichlet
   chdg::mark_boundary( mesh, 0, []( const Point& ) { return true; } );

   // Use triangles if the config file says so
   if ( element_type == "simplex" )
      MeshTools::Modification::all_tri( mesh );

   // Mesh Refinement object
   MeshRefinement mesh_refinement( mesh );
   mesh_refinement.refine_fraction() = refine_fraction;
   mesh_refinement.coarsen_fraction() = coarsen_fraction;
   mesh_refinement.max_h_level() = max_h_level;

   // Do uniform refinement
   for ( unsigned int rstep = 0; rstep < uniform_refinement_steps; rstep++ )
      mesh_refinement.uniformly_refine( 1 );

   // Crate an equation system object
   EquationSystems equation_system( mesh );

   // Set parameters for the equation system and the solver
   equation_system.parameters.set< Real >( "linear solver tolerance" ) = TOLERANCE * TOLERANCE;
   equation_system.parameters.set< unsigned int >( "linear solver maximum iterations" ) = 1000;
   equation_system.parameters.set< Real >( "penalty" ) = penalty;
   equation_system.parameters.set< bool >( "singularity" ) = singularity;
   equation_system.parameters.set< std::string >( "refinement" ) = refinement_type;

   // Create a system named ellipticdg
   auto& ellipticdg_system = equation_system.add_system< LinearImplicitSystem >( "EllipticDG" );

   // Add a variable "u" to "ellipticdg" using the p_order specified in the config file
   if ( on_command_line( "element_type" ) )
   {
      std::string fe_str = command_line_value( std::string( "element_type" ), std::string( "MONOMIAL" ) );

      libmesh_error_msg_if( fe_str != "MONOMIAL" || fe_str != "XYZ",
                            "Error: This example must be run with MONOMIAL or XYZ element types." );

      ellipticdg_system.add_variable( "u", p_order, Utility::string_to_enum< FEFamily >( fe_str ) );
   }
   else
      ellipticdg_system.add_variable( "u", p_order, MONOMIAL );

   chdg::integrator::EllipticIntegrator elliptic_integrator( ellipticdg_system );
   elliptic_integrator.set_trial_function( 0 );
   elliptic_integrator.set_test_function( 0 );
   elliptic_integrator.set_dirichlet_boundary_id( 0 );
   elliptic_integrator.set_dirichlet_boundary_value(
       [&]( const Point& p ) { return exact_solution( p, equation_system.parameters, "", "" ); } );
   elliptic_integrator.set_eps( +1 );
   elliptic_integrator.set_penalty( +1e-1 );

   ellipticdg_system.attach_assemble_object( elliptic_integrator );

   // Initialize the data structures for the equation system
   equation_system.init();

   // Construct ExactSolution object and attach solution functions
   ExactSolution exact_sol( equation_system );
   exact_sol.attach_exact_value( exact_solution );
   exact_sol.attach_exact_deriv( exact_derivative );

   // A refinement loop.
   for ( unsigned int rstep = 0; rstep < adaptive_refinement_steps; ++rstep )
   {
      libMesh::out << "  Beginning Solve " << rstep << std::endl;
      libMesh::out << "Number of elements: " << mesh.n_elem() << std::endl;

      // Solve the system
      ellipticdg_system.solve();

      libMesh::out << "System has: " << equation_system.n_active_dofs() << " degrees of freedom." << std::endl;

      libMesh::out << "Linear solver converged at step: " << ellipticdg_system.n_linear_iterations()
                   << ", final residual: " << ellipticdg_system.final_linear_residual() << std::endl;

      // Compute the error
      exact_sol.compute_error( "EllipticDG", "u" );

      // Print out the error values
      libMesh::out << "L2-Error is: " << exact_sol.l2_error( "EllipticDG", "u" ) << std::endl;

      // Possibly refine the mesh
      if ( rstep + 1 < adaptive_refinement_steps )
      {
         // The ErrorVector is a particular StatisticsVector
         // for computing error information on a finite element mesh.
         ErrorVector error;

         // The discontinuity error estimator
         // evaluate the jump of the solution
         // on elements faces
         DiscontinuityMeasure error_estimator;
         error_estimator.estimate_error( ellipticdg_system, error );

         // Take the error in error and decide which elements will be coarsened or refined
         mesh_refinement.flag_elements_by_error_fraction( error );
         if ( refinement_type == "p" )
            mesh_refinement.switch_h_to_p_refinement();
         if ( refinement_type == "hp" )
            mesh_refinement.add_p_to_h_refinement();

         // Refine and coarsen the flagged elements
         mesh_refinement.refine_and_coarsen_elements();
         equation_system.reinit();
      }
   }

   // Write out the solution
   // After solving the system write the solution
   // to a ExodusII-formatted plot file.
#ifdef LIBMESH_HAVE_EXODUS_API
   ExodusII_IO( mesh ).write_discontinuous_exodusII( "lshaped_dg.e", equation_system );
#endif

#endif // #ifndef LIBMESH_ENABLE_AMR

   // All done.
   return 0;
}
