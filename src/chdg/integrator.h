#pragma once

#include "libmesh/libmesh.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/transient_system.h"

namespace chdg {
namespace integrator {

namespace lm = libMesh;

/// Assembles the bilinear and linear form for
///     - div( kappa grad(phi) ) * dx
/// with a discontinuous Galerkin discretization, where
/// - tau is a scaling factor for the bilinear form (for instance time),
/// - kappa is a scalar function and
/// - phi is the solution of our bilinear form.
/// The corresponding (bi-)linear form with test function psi is
///     a(phi, psi) = tau * ( + kappa * grad(phi) * grad(psi) * dx
///                           - avg(kappa * grad(phi)) * n * jump(psi) * dS
///                           + eps * avg(kappa * grad(psi)) * n * jump(phi) * dS
///                           + penalty / h^beta jump(phi)*jump(psi) * dS )
///     l(psi) = + tau * ( + eps * kappa * grad(psi) * n * g_dir * ds_dir
///                        + penalty / h^beta psi * g_dir * ds_dir
///                        + psi * g_neu * ds_neu )
/// where ds_dir and ds_neu denote the dirichlet and neumann boundaries
/// and g_dir and g_neu are the dirichlet and neumann boundary values.
/// For eps = +1 we get the NIPG, for
///     eps = -1 we get the SIPG and for
///     eps =  0 we get the IIPG.
class EllipticIntegrator : public lm::System::Assembly
{
 public:
   explicit EllipticIntegrator( lm::LinearImplicitSystem& sys );

   using BoundaryFunction = std::function< void( const std::vector< lm::Point >&, std::vector< lm::Real >& values ) >;

   using CoefficientFunction =
       std::function< void( const lm::Elem*, const std::vector< lm::Point >&, std::vector< lm::Real >& ) >;

   void set_trial_function( unsigned int num ) { trial_function_ = num; }
   void set_test_function( unsigned int num ) { test_function_ = num; }

   void set_tau( const double tau ) { tau_ = tau; }

   void set_penalty( double penalty ) { penalty_ = penalty; }
   void set_eps( double eps ) { eps_ = eps; }
   void set_neumann_boundary_value( const BoundaryFunction& nbv ) { neumann_boundary_value_ = nbv; }
   void set_dirichlet_boundary_value( const BoundaryFunction& dbv ) { dirichlet_boundary_value_ = dbv; }
   void set_dirichlet_boundary_id( const lm::boundary_id_type& id ) { dirichlet_boundary_id_ = id; }

   void set_kappa( CoefficientFunction& kappa ) { kappa_ = kappa; }

   void assemble() override;
   void assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const;

 private:
   void assemble_elements( lm::SparseMatrix< lm::Real >& K ) const;
   void assemble_faces( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const;

 private:
   lm::LinearImplicitSystem& sys_;
   unsigned int trial_function_;
   unsigned int test_function_;

   BoundaryFunction neumann_boundary_value_;
   BoundaryFunction dirichlet_boundary_value_;

   CoefficientFunction kappa_;

   /// Boundary id marking the Dirichlet boundary.
   /// All other ids are interpreted as Neumann boundaries.
   lm::boundary_id_type dirichlet_boundary_id_;

   lm::Real penalty_;
   lm::Real eps_;

   double tau_;
};

/// Assembles the bilinear and linear form for
///     tau * div( vel * phi ) * dx
/// with a discontinuous Galerkin discretization, where
/// - tau is a scaling factor for the bilinear form (for instance time),
/// - vel is a given velocity and
/// - phi is the solution of our bilinear form.
/// The corresponding (bi-)linear form with test function psi is
///     a(phi, psi) = tau * ( - phi * vel * grad(psi) * dx
///                           + phi * vel * n * jump(psi) * dS
///                           + phi * vel * n * psi * ds_out )
///     l(psi) = - tau * phi_in * vel * n * psi * ds_in
/// where ds_in and ds_out denote the inflow and outflow boundaries
/// and phi_in is the inflow boundary value.
class AdvectionIntegrator : public lm::System::Assembly
{
 public:
   explicit AdvectionIntegrator( lm::LinearImplicitSystem& sys );

   using BoundaryFunction = std::function< lm::Number( const lm::Point& ) >;
   using VelocityField = std::function< lm::Point( const lm::Point& ) >;

   void set_trial_function( unsigned int num ) { trial_function_ = num; }
   void set_test_function( unsigned int num ) { test_function_ = num; }

   void set_tau( const double tau ) { tau_ = tau; }

   void set_velocity_field( VelocityField& velocity_field ) { velocity_field_ = velocity_field; }

   void set_inflow_boundary_value( const BoundaryFunction& dbv ) { inflow_boundary_value_ = dbv; }

   void assemble() override;
   void assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const;

 private:
   void assemble_elements( lm::SparseMatrix< lm::Real >& K ) const;
   void assemble_faces( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const;

 private:
   lm::LinearImplicitSystem& sys_;
   unsigned int trial_function_;
   unsigned int test_function_;

   std::function< lm::Number( const lm::Point& ) > inflow_boundary_value_;

   VelocityField velocity_field_;

   double tau_;
};

class MassLFIntegrator : public lm::System::Assembly
{
 public:
   explicit MassLFIntegrator( lm::LinearImplicitSystem& sys, lm::NumericVector< lm::Real >& u );

   void set_trial_function( unsigned int num ) { trial_function_ = num; }
   void set_test_function( unsigned int num ) { test_function_ = num; }

   void assemble() override;
   void assemble( lm::NumericVector< lm::Real >& F ) const;

 private:
   lm::LinearImplicitSystem& sys_;
   unsigned int trial_function_;
   unsigned int test_function_;

   lm::NumericVector< lm::Real >& u_;
};

class MassIntegrator : public lm::System::Assembly
{
 public:
   explicit MassIntegrator( lm::LinearImplicitSystem& sys );

   void set_trial_function( unsigned int num ) { trial_function_ = num; }
   void set_test_function( unsigned int num ) { test_function_ = num; }

   void assemble() override;
   void assemble( lm::SparseMatrix< lm::Real >& K ) const;

 private:
   lm::LinearImplicitSystem& sys_;
   unsigned int trial_function_;
   unsigned int test_function_;
};

class TransientImplicitAdvectionIntegrator : public lm::System::Assembly
{
 public:
   explicit TransientImplicitAdvectionIntegrator( lm::TransientLinearImplicitSystem& sys );

   void set_trial_function( unsigned int num );
   void set_test_function( unsigned int num );

   void set_dt( double dt );

   void assemble() override;
   void assemble( lm::SparseMatrix< lm::Real >& K, lm::NumericVector< lm::Real >& F ) const;

   /// Returns the advection integrator, which allows setting boundary conditions.
   integrator::AdvectionIntegrator& get_advection() { return advection_; }

 private:
   lm::TransientLinearImplicitSystem& sys_;

   integrator::AdvectionIntegrator advection_;
   integrator::MassIntegrator mass_;
   integrator::MassLFIntegrator mass_lf_;
};

} // namespace integrator
} // namespace chdg
