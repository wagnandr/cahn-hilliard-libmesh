#pragma once

#include <vector>

namespace libMesh {
class Point;
}

namespace chdg {

class LocalCellSystem;
class LocalInteriorFaceSystem;
class CellAssemblyData;
class FaceAssemblyData;

namespace forms {
namespace diffusion {

void assemble_cell( LocalCellSystem& local_system,
                    const CellAssemblyData& d,
                    const std::vector< double >& kappa,
                    const double& tau );

void assemble_face( LocalInteriorFaceSystem& local_system,
                    const FaceAssemblyData& d,
                    const std::vector< double >& kappa_e,
                    const std::vector< double >& kappa_n,
                    const double& tau,
                    const double& eps,
                    const double& penalty,
                    const double& beta );

} // namespace diffusion

namespace advection {

void assemble_face( LocalInteriorFaceSystem& local_system,
                    const FaceAssemblyData& d,
                    const std::vector< libMesh::Point >& velocity,
                    const double& tau );

}
} // namespace forms
} // namespace chdg
