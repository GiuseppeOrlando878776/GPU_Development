/*--- Author: Giuseppe Orlando, 2023 ---*/

// First include the necessary files from the deal.II library known from the
// previous tutorials.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

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

// The following ones include the data structures for the
// implementation of matrix-free methods on GPU:
#include <deal.II/base/cuda.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>

#include "runtime_parameters.h"

// As usual, we enclose everything into a namespace of its own:
//
namespace Step64 {
  using namespace dealii;

  // @sect{Class <code>HelmholtzOperatorQuad</code>}

  // The class `HelmholtzOperatorQuad` implements the evaluation of
  // the Helmholtz operator at each quadrature point. It uses a
  // similar mechanism as the MatrixFree framework introduced in
  // step-37. In contrast to there, the actual quadrature point
  // index is treated implicitly by converting the current thread
  // index. As before, the functions of this class need to run on
  // the device, so need to be marked as `__device__` for the
  // compiler.
  //
  template <int dim, int fe_degree>
  class HelmholtzOperatorQuad {
  public:
    __device__ HelmholtzOperatorQuad() {}

    __device__ void operator()(CUDAWrappers::FEEvaluation<dim, fe_degree>* fe_eval) const;
  };


  // The class `HelmholtzOperatorQuad` implements the evaluation of
  // the mass matrix at each quadrature point. The functions of this class
  // need to run on the device, so need to be marked as `__device__` for the
  // compiler.
  //
  template <int dim, int fe_degree>
  __device__ void HelmholtzOperatorQuad<dim, fe_degree>::operator()(CUDAWrappers::FEEvaluation<dim, fe_degree>* fe_eval) const {
    fe_eval->submit_value(fe_eval->get_value());
  }


  // @sect{Class <code>LocalHelmholtzOperator</code>}

  // Finally, we need to define a class that implements the whole operator
  // evaluation that corresponds to a matrix-vector product in matrix-based
  // approaches.
  //
  template <int dim, int fe_degree, int n_q_points_1d>
  class LocalHelmholtzOperator {
  public:
    LocalHelmholtzOperator() {}

    __device__ void operator()(const unsigned int                                          cell,
                               const typename CUDAWrappers::MatrixFree<dim, double>::Data* gpu_data,
                               CUDAWrappers::SharedData<dim, double>*                      shared_data,
                               const double*                                               src,
                               double*                                                     dst) const;

    // Again, the CUDAWrappers::MatrixFree object doesn't know about the number
    // of degrees of freedom and the number of quadrature points so we need
    // to store these for index calculations in the call operator.
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(n_dofs_1d, dim);
    static const unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);
  };


  // This is the call operator that performs the Helmholtz operator evaluation
  // on a given cell similar to the MatrixFree framework on the CPU.
  // In particular, we need access to both values and gradients of the source
  // vector and we write value and gradient information to the destination
  // vector.
  template <int dim, int fe_degree, int n_q_points_1d>
  __device__ void LocalHelmholtzOperator<dim, fe_degree, n_q_points_1d>::
  operator()(const unsigned int                                          cell,
             const typename CUDAWrappers::MatrixFree<dim, double>::Data* gpu_data,
             CUDAWrappers::SharedData<dim, double>*                      shared_data,
             const double*                                               src,
             double*                                                     dst) const {
    CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, double> fe_eval(cell, gpu_data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, false); /*--- This two-stage procedure is necessary because gather_evaluate has not been defined ---*/

    fe_eval.apply_for_each_quad_point(HelmholtzOperatorQuad<dim, fe_degree>());

    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(dst); /*--- This two-stage procedure is necessary because integrate_scatter has not been defined ---*/
  }


  // @sect{Class <code>HelmholtzOperator</code>}

  // The `HelmholtzOperator` class acts as wrapper for
  // `LocalHelmholtzOperator` defining an interface that can be used
  // with linear solvers like SolverCG. In particular, like every
  // class that implements the interface of a linear operator, it
  // needs to have a `vmult()` function that performs the action of
  // the linear operator on a source vector.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping = 1>
  class HelmholtzOperator {
  public:
    HelmholtzOperator(const DoFHandler<dim>&           dof_handler,
                      const AffineConstraints<double>& constraints);

    void initialize_dof_vector(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& vec) const;

    void vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>&       dst,
               const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& src) const;

  private:
    CUDAWrappers::MatrixFree<dim, double> mf_data; /*--- Notice that this class cannot be dervied from MatrixFreeOperators::Base
                                                         because we need CUDAWrappers::MatrixFree instance ---*/
  };


  // The following is the implementation of the constructor of this
  // class. In the first part, we initialize the `mf_data` member
  // variable that is going to provide us with the necessary
  // information when evaluating the operator.
  //
  // In the second half, we need to store the value of the coefficient
  // for each quadrature point in every active, locally owned cell.
  // We can ask the parallel triangulation for the number of active, locally
  // owned cells but only have a DoFHandler object at hand. Since
  // DoFHandler::get_triangulation() returns a Triangulation object, not a
  // parallel::TriangulationBase object, we have to downcast the return value.
  // This is safe to do here because we know that the triangulation is a
  // parallel:distributed::Triangulation object in fact.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  HelmholtzOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  HelmholtzOperator(const DoFHandler<dim>&           dof_handler,
                    const AffineConstraints<double>& constraints) {
    MappingQGeneric<dim> mapping(degree_mapping); /*--- Mapping ---*/

    typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData additional_data; /*--- Additional data with flags to be initialized ---*/
    additional_data.mapping_update_flags = update_values | update_JxW_values | update_quadrature_points;

    const QGauss<1> quad(n_q_points_1d); /*--- Quadrature formula ---*/

    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data); /*--- Reinit the matrix free structure ---*/
  }


  // Auxiliary function to initialize vectors since we cannot dervei from MatrixFreeOperators and, therefore,
  // we cannot use a MatrixFree to directly initialize the HelmholtzOperator and the corresponding vectors.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  void HelmholtzOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  initialize_dof_vector(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& vec) const {
    mf_data.initialize_dof_vector(vec);
  }


  // The key step then is to use all of the previous classes to loop over
  // all cells to perform the matrix-vector product. We implement this
  // in the next function.
  //
  // When applying the Helmholtz operator, we have to be careful to handle
  // boundary conditions correctly. Since the local operator doesn't know about
  // constraints, we have to copy the correct values from the source to the
  // destination vector afterwards.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  void HelmholtzOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>&       dst,
        const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& src) const {
    dst = 0;

    LocalHelmholtzOperator<dim, fe_degree, fe_degree + 1> helmholtz_operator;
    mf_data.cell_loop(helmholtz_operator, src, dst);
    /*--- The cell_loop needs as first input argument a Functor with a __device__ void operator().
          Hence, we need a class for the local action and a simple routine seems not to be sufficient because of the operator() request. ---*/

    mf_data.copy_constrained_values(src, dst);
  }


  // @sect{Class <code>HelmholtzProblem</code>}

  // This is the main class of this program. It defines the usual
  // framework we use for tutorial programs. The only point worth
  // commenting on is the `solve()` function and the choice of vector
  // types.
  //
  template <int dim, int fe_degree>
  class HelmholtzProblem {
  public:
    HelmholtzProblem(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose, const unsigned int output_interval);

  protected:
    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;

    DoFHandler<dim> dof_handler;

    // Since all the operations in the `solve()` function are executed on the
    // graphics card, it is necessary for the vectors used to store their values
    // on the GPU as well. LinearAlgebra::distributed::Vector can be told which
    // memory space to use. There is also LinearAlgebra::CUDAWrappers::Vector
    // that always uses GPU memory storage but doesn't work with MPI. It might
    // be worth noticing that the communication between different MPI processes
    // can be improved if the MPI implementation is CUDA-aware and the configure
    // flag `DEAL_II_MPI_WITH_CUDA_SUPPORT` is enabled. (The value of this
    // flag needs to be set at the time you call `cmake` when installing
    // deal.II.)
    //
    // In addition, we also keep a solution vector with CPU storage such that we
    // can view and display the solution as usual.
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution_dev;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host;

  private:
    void create_triangulation(const unsigned int n_refines);

    void setup_system();

    void assemble_rhs();

    void solve();

    void output_results() const;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
    double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

    std::string  saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

    /*--- Now we declare a bunch of variables for text output ---*/
    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::unique_ptr<HelmholtzOperator<dim, fe_degree, fe_degree + 1>> system_matrix_dev;
  };


  // The implementation of all the remaining functions of this class apart from
  // `Helmholtzproblem::solve()` doesn't contain anything new and we won't
  // further comment much on the overall approach.
  //
  template <int dim, int fe_degree>
  HelmholtzProblem<dim, fe_degree>::HelmholtzProblem(RunTimeParameters::Data_Storage& data) :
    triangulation(MPI_COMM_WORLD),
    fe(fe_degree),
    dof_handler(triangulation),
    max_its(data.max_iterations),
    eps(data.eps),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
    Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times) {
      create_triangulation(data.n_global_refines);
      setup_system();
    }


  // Build the domain
  //
  template<int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::create_triangulation(const unsigned int n_refines) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

    triangulation.refine_global(n_refines);
  }


  // Setup the system
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::setup_system() {
    dof_handler.distribute_dofs(fe);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    system_matrix_dev.reset(new HelmholtzOperator<dim, fe_degree, fe_degree + 1>(dof_handler, constraints));
    system_matrix_dev->initialize_dof_vector(solution_dev);
    system_matrix_dev->initialize_dof_vector(system_rhs_dev);
  }


  // Unlike programs such as step-4 or step-6, we will not have to
  // assemble the whole linear system but only the right hand side
  // vector. This looks in essence like we did in step-4, for example,
  // but we have to pay attention to using the right constraints
  // object when copying local contributions into the global
  // vector.
  //
  // At the end of the function, we can't directly copy the values
  // from the host to the device but need to use an intermediate
  // object of type LinearAlgebra::ReadWriteVector to construct the
  // correct communication pattern necessary.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::assemble_rhs() {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(locally_owned_dofs,
                                                                                  locally_relevant_dofs,
                                                                                  MPI_COMM_WORLD);
    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        cell_rhs = 0;

        fe_values.reinit(cell);

        for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
          for(unsigned int i = 0; i < dofs_per_cell; ++i) {
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *
                            fe_values.JxW(q_index));
          }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               local_dof_indices,
                                               system_rhs_host);
      }
    }
    system_rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import(rw_vector, VectorOperation::insert);
  }


  // This solve() function finally contains the calls to the new classes
  // previously discussed. Here we don't use any preconditioner, i.e.,
  // precondition by the identity matrix, to focus just on the peculiarities of
  // the CUDAWrappers::MatrixFree framework.
  //
  // After solving the linear system in the first part of the function, we
  // copy the solution from the device to the host to be able to view its
  // values and display it in `output_results()`. This transfer works the same
  // as at the end of the previous function.
  //
  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::solve() {
    PreconditionIdentity preconditioner;

    SolverControl solver_control(max_its, eps*system_rhs_dev.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>> cg(solver_control);
    cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

    pcout << "  Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(solution_dev, VectorOperation::insert);
    solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(solution_host);
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
  void HelmholtzProblem<dim, fe_degree>::output_results() const {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    solution_host.update_ghost_values();
    data_out.add_data_vector(solution_host, "solution");

    data_out.build_patches();

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    const std::string output_file = "./" + saving_dir + "/solution.vtu";
    data_out.write_vtu_in_parallel(output_file, MPI_COMM_WORLD);

    Vector<float> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution_host,
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
  void HelmholtzProblem<dim, fe_degree>::run(const bool verbose, const unsigned int output_interval) {
    assemble_rhs();
    solve();
    output_results();
  }

} // namespace Step64


// @sect{The <code>main()</code> function}

// Finally for the `main()` function.  By default, all the MPI ranks
// will try to access the device with number 0, which we assume to be
// the GPU device associated with the CPU on which a particular MPI
// rank runs. This works, but if we are running with MPI support it
// may be that multiple MPI processes are running on the same machine
// (for example, one per CPU core) and then they would all want to
// access the same GPU on that machine. If there is only one GPU in
// the machine, there is nothing we can do about it: All MPI ranks on
// that machine need to share it. But if there are more than one GPU,
// then it is better to address different graphic cards for different
// processes. The choice below is based on the MPI process id by
// assigning GPUs round robin to GPU ranks. (To work correctly, this
// scheme assumes that the MPI ranks on one machine are
// consecutive. If that were not the case, then the rank-GPU
// association may just not be optimal.) To make this work, MPI needs
// to be initialized before using this function.
//
int main(int argc, char *argv[]) {
  try
  {
    using namespace Step64;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    int         n_devices       = 0;
    cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
    AssertCuda(cuda_error_code);
    const unsigned int my_mpi_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const int device_id = my_mpi_id % n_devices;
    cuda_error_code     = cudaSetDevice(device_id);
    AssertCuda(cuda_error_code);

    deallog.depth_console(data.verbose && my_mpi_id == 0 ? 2 : 0);

    HelmholtzProblem<3, 3> helmholtz_problem(data);
    helmholtz_problem.run(data.verbose, data.output_interval);
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
