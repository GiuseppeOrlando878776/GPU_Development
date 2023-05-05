/* Author: Giuseppe Orlando, 2023. */

// @sect{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// This is the class that implements the discretization
//
namespace Advection {
  using namespace dealii;

  // @sect{ <code>AdvectionOperator::AdvectionOperator</code> }
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  class AdvectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    AdvectionOperator(); /*--- Default constructor ---*/

    AdvectionOperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_HYPERBOLIC_stage(const unsigned int stage); /*--- Setter of the IMEX stage. ---*/

    void vmult_rhs_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs. ---*/

    virtual void compute_diagonal() override; /*--- Overriden function to compute the diagonal. ---*/

  protected:
    mutable Tensor<1, dim, VectorizedArray<Number>> velocity;

    EquationData::Velocity<dim> u; /*--- Advecting field ---*/

    double       dt; /*--- Time step. ---*/

    unsigned int HYPERBOLIC_stage; /*--- Flag for the time discretization scheme stage ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    /*--- Assembler functions for the rhs. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_update(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_update(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const Vec&                                   src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the matrix. For compatibilty conditions,
          also face and boundary contributions have to be defined, even though they are empty. ---*/
    void assemble_diagonal_cell_term_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const {}
    void assemble_diagonal_boundary_term_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const {}
  };


  // Default constructor
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  AdvectionOperator(): MatrixFreeOperators::Base<dim, Vec>(), u(), dt(), HYPERBOLIC_stage(1) {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  AdvectionOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                            u(data.initial_time), dt(data.dt), HYPERBOLIC_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  set_HYPERBOLIC_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    HYPERBOLIC_stage = stage;
  }


  // Assemble rhs cell term for the advected variable update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_rhs_cell_term_update(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, 1),
                                                           phi_rho_prev(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_prev.reinit(cell);
      phi_rho_prev.gather_evaluate(src[0], EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& point_vectorized = phi.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
          Point<dim> point;
          for(unsigned int d = 0; d < dim; ++d) {
            point[d] = point_vectorized[d][v];
          }
          for(unsigned int d = 0; d < dim; ++d) {
            velocity[d][v] = u.value(point, d);
          }
        }

        phi.submit_value(phi_rho_prev.get_value(q), q);
        phi.submit_gradient(dt*phi_rho_prev.get_value(q)*velocity, q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the advective variable update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_rhs_face_term_update(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true, 1),
                                                               phi_m(data, false, 1),
                                                               phi_rho_prev_p(data, true, 1),
                                                               phi_rho_prev_m(data, false, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_prev_p.reinit(face);
      phi_rho_prev_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_prev_m.reinit(face);
      phi_rho_prev_m.gather_evaluate(src[0], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& point_vectorized = phi_p.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
          Point<dim> point;
          for(unsigned int d = 0; d < dim; ++d) {
            point[d] = point_vectorized[d][v];
          }
          for(unsigned int d = 0; d < dim; ++d) {
            velocity[d][v] = u.value(point, d);
          }
        }

        const auto& n_plus        = phi_p.get_normal_vector(q);

        const auto& avg_flux      = 0.5*(phi_rho_prev_p.get_value(q)*velocity +
                                         phi_rho_prev_m.get_value(q)*velocity);
        const auto  lambda_prev   = std::max(std::abs(scalar_product(velocity, n_plus)),
                                             std::abs(scalar_product(velocity, n_plus)));
        const auto& jump_rho_prev = phi_rho_prev_p.get_value(q) - phi_rho_prev_m.get_value(q);

        phi_p.submit_value(-dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_prev*jump_rho_prev), q);
        phi_m.submit_value(dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_prev*jump_rho_prev), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  vmult_rhs_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&AdvectionOperator::assemble_rhs_cell_term_update,
                     &AdvectionOperator::assemble_rhs_face_term_update,
                     &AdvectionOperator::assemble_rhs_boundary_term_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_cell_term_update(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, 1);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q); /*--- Here we need to assemble just a mass matrix,
                                                     so we simply test against the test fuction, the 'src' vector ---*/
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    this->data->cell_loop(&AdvectionOperator::assemble_cell_term_update,
                          this, dst, src, false);
  }


  // Assemble diagonal cell term for the rho projection
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_diagonal_cell_term_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }
        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, Vec>::
  compute_diagonal() {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    this->data->initialize_dof_vector(inverse_diagonal, 1);
    Vec dummy;
    dummy.reinit(inverse_diagonal.local_size());

    this->data->loop(&AdvectionOperator::assemble_diagonal_cell_term_update,
                     &AdvectionOperator::assemble_diagonal_face_term_update,
                     &AdvectionOperator::assemble_diagonal_boundary_term_update,
                     this, inverse_diagonal, dummy, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }
}
