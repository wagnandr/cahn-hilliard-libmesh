#pragma once

#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/quadrature_gauss.h"

namespace chdg {

namespace lm = libMesh;

inline std::unique_ptr< lm::FEBase > build_fe_with_qrule( const unsigned int dim, const lm::FEType& fe_type, lm::QBase& qrule )
{
   std::unique_ptr< lm::FEBase > fe = lm::FEBase::build( dim, fe_type );
   fe->attach_quadrature_rule( &qrule );
   return fe;
}

class CellAssemblyData
{
 public:
   CellAssemblyData( const unsigned int dim, const lm::FEType& fe_type_trial, const lm::FEType& fe_type_test )
   : qrule( dim, fe_type_trial.default_quadrature_order() )
   , fe( build_fe_with_qrule( dim, fe_type_trial, qrule ) )
   , phi( fe->get_phi() )
   , dphi( fe->get_dphi() )
   , psi( fe->get_phi() )
   , dpsi( fe->get_dphi() )
   , xyz( fe->get_xyz() )
   , JxW( fe->get_JxW() )
   , element( nullptr )
   {
      if ( fe_type_trial != fe_type_test )
         throw std::runtime_error( "Different elements for trial and test functions not supported yet." );
   }

 private:
   lm::QGauss qrule;

   std::unique_ptr< lm::FEBase > fe;

 public:
   const std::vector< std::vector< lm::Real > >& phi;
   const std::vector< std::vector< lm::RealGradient > >& dphi;
   const std::vector< std::vector< lm::Real > >& psi;
   const std::vector< std::vector< lm::RealGradient > >& dpsi;

   const std::vector< lm::Real >& JxW;
   const std::vector< lm::Point >& xyz;

   /*! @brief Pointer to the element to which this assembly data belongs. */
   const lm::Elem* element;

 public:
   /// Computes the values of the shape functions for cell.
   void init( const lm::Elem* elem )
   {
      fe->reinit( elem );
      element = elem;
   }

   unsigned int get_order() const { return fe->get_order(); }

   unsigned int num_shape_functions() const { return fe->n_shape_functions(); }

   unsigned int num_quad_points() const { return qrule.n_points(); }
};

class FaceAssemblyData
{
 public:
   FaceAssemblyData( const unsigned int dim, const lm::FEType& fe_type_trial, const lm::FEType& fe_type_test )
   : qrule( dim - 1, fe_type_trial.default_quadrature_order() )
   , fe_e( build_fe_with_qrule( dim, fe_type_trial, qrule ) )
   , fe_n( build_fe_with_qrule( dim, fe_type_trial, qrule ) )
   , is_exterior_( false )
   , face( nullptr )
   , phi_e( fe_e->get_phi() )
   , dphi_e( fe_e->get_dphi() )
   , psi_e( fe_e->get_phi() )
   , dpsi_e( fe_e->get_dphi() )
   , phi_n( fe_n->get_phi() )
   , dphi_n( fe_n->get_dphi() )
   , psi_n( fe_n->get_phi() )
   , dpsi_n( fe_n->get_dphi() )
   , xyz_e( fe_e->get_xyz() )
   , JxW( fe_e->get_JxW() )
   , n_e( fe_e->get_normals() )
   , element( nullptr )
   , neighbor( nullptr )
   {
      if ( fe_type_trial != fe_type_test )
         throw std::runtime_error( "Different elements for trial and test functions not supported yet." );
   }

 private:
   lm::QGauss qrule;

   std::unique_ptr< lm::FEBase > fe_e;
   std::unique_ptr< lm::FEBase > fe_n;

   bool is_exterior_;

   std::unique_ptr< const lm::Elem > face;

 public:
   const std::vector< std::vector< lm::Real > >& phi_e;
   const std::vector< std::vector< lm::RealGradient > >& dphi_e;
   const std::vector< std::vector< lm::Real > >& psi_e;
   const std::vector< std::vector< lm::RealGradient > >& dpsi_e;

   const std::vector< lm::Real >& JxW;
   const std::vector< lm::Point >& n_e;
   const std::vector< lm::Point >& xyz_e;

   const std::vector< std::vector< lm::Real > >& phi_n;
   const std::vector< std::vector< lm::RealGradient > >& dphi_n;
   const std::vector< std::vector< lm::Real > >& psi_n;
   const std::vector< std::vector< lm::RealGradient > >& dpsi_n;

   const lm::Elem* element;
   const lm::Elem* neighbor;

 public:
   /// Computes the values of the shape functions for a face and its neighbor.
   void init( const lm::Elem* elem, const unsigned short side )
   {
      element = elem;
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

   unsigned int get_order_e() const { return fe_e->get_order(); }

   unsigned int get_order_n() const { return fe_n->get_order(); }

   unsigned int num_shape_functions_e() const { return fe_e->n_shape_functions(); }

   unsigned int num_shape_functions_n() const { return fe_n->n_shape_functions(); }

   bool is_exterior() const { return is_exterior_; };

   unsigned int num_quad_points() const { return qrule.n_points(); }

   double get_h() const { return face->volume(); }

 private:
   void init_exterior( const lm::Elem* elem, const unsigned short side ) { fe_e->reinit( elem, side ); }

   void init_interior( const lm::Elem* elem, const unsigned short side )
   {
      // initialize the element itself
      fe_e->reinit( elem, side );

      neighbor = elem->neighbor_ptr( side );

      // we now map the quadrature points elem to its neighbor
      auto qrule_face_points = fe_e->get_xyz();
      std::vector< lm::Point > qrule_face_neighbor_points;
      lm::FEMap::inverse_map( elem->dim(), neighbor, qrule_face_points, qrule_face_neighbor_points );
      fe_n->reinit( neighbor, &qrule_face_neighbor_points );

      face = elem->build_side_ptr( side );
   }
};

} // namespace chdg