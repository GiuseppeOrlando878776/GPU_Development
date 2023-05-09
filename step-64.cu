/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Bruno Turcksin, Daniel Arndt, Oak Ridge National Laboratory, 2019
 */

// First include the necessary files from the deal.II library known from the
// previous tutorials.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// The following ones include the data structures for the
// implementation of matrix-free methods on GPU:
#include <deal.II/base/cuda.h>

#include <deal.II/lac/cuda_sparse_matrix.h>

#include <fstream>

// As usual, we enclose everything into a namespace of its own:
//
namespace Step64 {
  using namespace dealii;

  // @sect3{Class <code>HelmholtzProblem</code>}

  // This is the main class of this program. It defines the usual
  // framework we use for tutorial programs. The only point worth
  // commenting on is the `solve()` function and the choice of vector
  // types.
  //
  template <int dim, int fe_degree>
  class HelmholtzProblem {
  public:
    HelmholtzProblem();

    void run();

  private:
    void setup_system();

    void assemble_system();

    void solve();

    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    // Since all the operations in the `solve()` function are executed on the
    // graphics card, it is necessary for the vectors used to store their values
    // on the GPU as well.
    //
    // In addition, we also keep a solution vector with CPU storage such that we
    // can view and display the solution as usual.
    CUDAWrappers::SparseMatrix<double> system_matrix_dev;
    SparsityPattern sparsity_pattern;

    LinearAlgebra::CUDAWrappers::Vector<double> solution_dev;
    LinearAlgebra::CUDAWrappers::Vector<double> system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> ghost_solution_host;

    ConditionalOStream pcout;
  };


  // The implementation of all the remaining functions of this class apart from
  // `Helmholtzproblem::solve()` doesn't contain anything new and we won't
  // further comment much on the overall approach.
  //
  template <int dim, int fe_degree>
  HelmholtzProblem<dim, fe_degree>::HelmholtzProblem() : triangulation(),
                                                         fe(fe_degree),
                                                         dof_handler(triangulation),
                                                         pcout(std::cout) {}


  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::setup_system() {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    solution_dev.reinit(dof_handler.n_dofs());
    system_rhs_dev.reinit(dof_handler.n_dofs());

    ghost_solution_host.reinit(dof_handler.n_dofs());
  }


  // We assemble now the matrix.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::assemble_system() {
    SparseMatrix<double> system_matrix_host;
    system_matrix_host.reinit(sparsity_pattern);

    Vector<double> system_rhs_host(dof_handler.n_dofs());

    const QGauss<dim> quadrature_formula(fe_degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        const double coeff_aux = fe_values.get_quadrature_points()[q_index].norm_square();
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            cell_matrix(i,j) += ((fe_values.shape_grad(i, q_index)*fe_values.shape_grad(j, q_index) +
                                 10.0/(0.05 + 2.0*coeff_aux)*fe_values.shape_value(i, q_index)*fe_values.shape_value(j, q_index))*
                                 fe_values.JxW(q_index));
          }

          cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *
                          fe_values.JxW(q_index));
        }
      }

      cell->get_dof_indices(local_dof_indices);

      MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices, cell_matrix, cell_rhs, true);

      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix_host,
                                             system_rhs_host);
    }
    system_matrix_host.compress(VectorOperation::add);
    system_rhs_host.compress(VectorOperation::add);

    Utilities::CUDA::Handle cuda_handle;
    system_matrix_dev.reinit(cuda_handle, system_matrix_host);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import(rw_vector, VectorOperation::insert);
  }


  // This solve() function finally contains the calls to the new classes
  // previously discussed. Here we don't use any preconditioner, i.e.,
  // precondition by the identity matrix, to focus just on the peculiarities of
  // the CUDAWrappers::MatrixFree framework. Of course, in a real application
  // the choice of a suitable preconditioner is crucial but we have at least the
  // same restrictions as in step-37 since matrix entries are computed on the
  // fly and not stored.
  //
  // After solving the linear system in the first part of the function, we
  // copy the solution from the device to the host to be able to view its
  // values and display it in `output_results()`. This transfer works the same
  // as at the end of the previous function.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::solve() {
    PreconditionIdentity preconditioner;

    SolverControl solver_control(system_rhs_dev.size(), 1e-12*system_rhs_dev.l2_norm());
    SolverCG<LinearAlgebra::CUDAWrappers::Vector<double>> cg(solver_control);
    cg.solve(system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

    pcout << "  Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(solution_dev, VectorOperation::insert);
    ghost_solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(ghost_solution_host);

    ghost_solution_host.update_ghost_values();
  }

  // The output results function is as usual since we have already copied the
  // values back from the GPU to the CPU.
  //
  // While we're already doing something with the function, we might
  // as well compute the $L_2$ norm of the solution. We do this by
  // calling VectorTools::integrate_difference(). That function is
  // meant to compute the error by evaluating the difference between
  // the numerical solution (given by a vector of values for the
  // degrees of freedom) and an object representing the exact
  // solution. But we can easily compute the $L_2$ norm of the
  // solution by passing in a zero function instead. That is, instead
  // of evaluating the error $\|u_h-u\|_{L_2(\Omega)}$, we are just
  // evaluating $\|u_h-0\|_{L_2(\Omega)}=\|u_h\|_{L_2(\Omega)}$
  // instead.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::output_results(const unsigned int cycle) const {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ghost_solution_host, "solution");
    data_out.build_patches();

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    std::ofstream output_file("./solution_" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output_file);

    Vector<float> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      Functions::ZeroFunction<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double global_norm = VectorTools::compute_global_error(triangulation,
                                                                 cellwise_norm,
                                                                 VectorTools::L2_norm);
    pcout << "  solution norm: " << global_norm << std::endl;
  }


  // There is nothing surprising in the `run()` function either. We simply
  // compute the solution on a series of (globally) refined meshes.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::run() {
    for(unsigned int cycle = 0; cycle < 7 - dim; ++cycle) {
      pcout << "Cycle " << cycle << std::endl;

      if(cycle == 0) {
        GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
      }
      triangulation.refine_global(1);

      setup_system();

      pcout << "   Number of active cells:       "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
      pcout << std::endl;
    }
  }
} // namespace Step64


// @sect3{The <code>main()</code> function}

// Finally for the `main()` function.  By default, all the MPI ranks
// will try to access the device with number 0, which we assume to be
// the GPU device associated with the CPU.
//
int main(int argc, char *argv[]) {
  try
  {
    using namespace Step64;

    int         n_devices       = 0;
    cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
    AssertCuda(cuda_error_code);
    cuda_error_code = cudaSetDevice(0);
    AssertCuda(cuda_error_code);

    HelmholtzProblem<3, 3> helmholtz_problem;
    helmholtz_problem.run();
  }
  catch(std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
