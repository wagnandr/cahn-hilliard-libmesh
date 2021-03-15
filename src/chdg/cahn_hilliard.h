#pragma once

#include "libmesh/dense_matrix.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/libmesh.h"
#include "libmesh/system.h"
#include "libmesh/transient_system.h"

namespace libMesh {
class DofMap;
class Elem;
} // namespace libMesh

namespace chdg {

namespace lm = libMesh;

class CellAssemblyData;
class FaceAssemblyData;

namespace util {

inline lm::Real cutoff( lm::Real value )
{
   if ( value > 1 )
      return 1;
   else if ( value < 0 )
      return 0;
   else
      return value;
}

// TODO: Regularize this!
inline lm::Real heavyside( lm::Real value )
{
   if ( value > 0 )
      return value;
   else
      return 0;
}

} // namespace util

class EvaluateParametersOnCell
{
 public:
   virtual ~EvaluateParametersOnCell() = default;

   virtual void init( const CellAssemblyData& data ) = 0;

   virtual void mobility( std::vector< lm::Real >& mobility ) const = 0;
   virtual void rhs_c( std::vector< lm::Real >& rhs ) const = 0;
   virtual void rhs_mu( std::vector< lm::Real >& rhs ) const = 0;
};

class EvaluateParametersOnFace
{
 public:
   virtual ~EvaluateParametersOnFace() = default;

   virtual void init( const FaceAssemblyData& data ) = 0;
   virtual void mobility( std::vector< lm::Real >& mobility_e, std::vector< lm::Real >& mobility_n ) const = 0;
};

/*! @brief Function type to evaluate the mobility and rhs on the quadrature points of a cell. */
class CahnHilliardLocalSystemData
{
 public:
   CahnHilliardLocalSystemData( unsigned int variable_number_c, unsigned int variable_number_mu );

   // local matrices
   lm::DenseMatrix< lm::Number > K;
   lm::DenseVector< lm::Number > F;

   // local submatrices
   lm::DenseSubMatrix< lm::Number > K_cc;
   lm::DenseSubMatrix< lm::Number > K_cmu;
   lm::DenseSubMatrix< lm::Number > K_muc;
   lm::DenseSubMatrix< lm::Number > K_mumu;
   lm::DenseSubVector< lm::Number > F_c;
   lm::DenseSubVector< lm::Number > F_mu;

   // dof indices
   std::vector< lm::dof_id_type > dof_indices;
   std::vector< lm::dof_id_type > dof_indices_c;
   std::vector< lm::dof_id_type > dof_indices_mu;

   void init( const lm::DofMap& dof_map, const lm::Elem& element );

 private:
   const unsigned int variable_number_c;
   const unsigned int variable_number_mu;
};

/*! @brief Function type to evaluate the mobility and rhs on the quadrature points of an interior face . */
class EvaluatorOnQuadraturePoints
{
 public:
   EvaluatorOnQuadraturePoints( const lm::TransientLinearImplicitSystem& sys, const std::string& name );

   void evaluate_old( const CellAssemblyData& data, std::vector< lm::Real >& values ) const;

   void evaluate_old( const FaceAssemblyData& data, std::vector< lm::Real >& values_e, std::vector< lm::Real >& values_n ) const;

 private:
   const lm::TransientLinearImplicitSystem& sys_;
   const unsigned int variable_number_;

   // Helper data structures for cells
   mutable std::vector< lm::dof_id_type > dof_indices;
   mutable std::vector< lm::Real > dof_values;

   // Helper data structures for interior faces
   mutable std::vector< lm::dof_id_type > dof_indices_e;
   mutable std::vector< lm::dof_id_type > dof_indices_n;
   mutable std::vector< lm::Real > dof_values_e;
   mutable std::vector< lm::Real > dof_values_n;
};

/*! @brief Assembly class for the Cahn-Hilliard equation. */
class CahnHilliardDGAssembly : public lm::System::Assembly
{
 public:
   explicit CahnHilliardDGAssembly( lm::TransientLinearImplicitSystem& sys );

   /*! @brief Assembly function. Overrides the default assembly function */
   void assemble() override;

   void set_evaluate_params_on_cell( std::shared_ptr< EvaluateParametersOnCell > );
   void set_evaluate_params_on_face( std::shared_ptr< EvaluateParametersOnFace > );

   void set_dt( double dt );
   void set_C_psi( double C_psi );
   void set_epsilon( double epsilon );

   enum class DGScheme
   {
      SIP,
      NIP,
      IIP
   };

   void set_scheme( DGScheme scheme );

   /*! @brief Sets the penalties terms for the laplacian of the concentration and potential variable. */
   void set_penalties( double penalty_c, double penalty_mu );

 private:
   double dt_;
   double C_psi_;
   double epsilon_;

   std::shared_ptr< EvaluateParametersOnCell > evaluate_params_on_cell_;
   std::shared_ptr< EvaluateParametersOnFace > evaluate_params_on_face_;

   EvaluatorOnQuadraturePoints c_evaluator;

   double eps_;
   double penalty_c_;
   double penalty_mu_;
   double beta_;

   lm::TransientLinearImplicitSystem& sys_;

 private:
   /*! @brief Assembly of the cell contributions. */
   void assemble_elements();

   /*! @brief Assembly of the interior face contributions. */
   void assemble_faces();

   /*! @brief Do the discontinuous terms get assembled? */
   bool use_dg() const;
};

} // namespace chdg