/*--- Author: Giuseppe Orlando, 2023 ---*/

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
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

// Include some other fudamental headers
#include <fstream>
#include <stdio.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// As usual, we enclose everything into a namespace of its own:
//
namespace AdvectionSolver {
  using namespace dealii;

  // @sect{Class <code>AdvectionOperatorQuad</code>}

  // The class `AdvectionOperatorQuad` implements the evaluation of
  // the mass matrix at each quadrature point. The functions of this class
  // need to run on the device, so need to be marked as `__device__` for the
  // compiler.
  //
  template <int dim, int fe_degree>
  class AdvectionOperatorQuad {
  public:
    __device__ AdvectionOperatorQuad() {}

    __device__ void operator()(CUDAWrappers::FEEvaluation<dim, fe_degree>* fe_eval) const;
  };


  // Since we are using an explicit time integration scheme, we just need to
  // assemble a mass matrix.
  //
  template <int dim, int fe_degree>
  __device__ void AdvectionOperatorQuad<dim, fe_degree>::operator()(CUDAWrappers::FEEvaluation<dim, fe_degree>* fe_eval) const {
    fe_eval->submit_value(fe_eval->get_value());
  }


  // @sect{Class <code>LocalAdvectionOperator</code>}

  // Finally, we need to define a class that implements the whole operator
  // evaluation that corresponds to a matrix-vector product in matrix-based
  // approaches.
  //
  template <int dim, int fe_degree, int n_q_points_1d>
  class LocalAdvectionOperator {
  public:
    LocalAdvectionOperator() {}

    __device__ void operator()(const unsigned int                                          cell,
                               const typename CUDAWrappers::MatrixFree<dim, double>::Data* gpu_data,
                               CUDAWrappers::SharedData<dim, double>*                      shared_data,
                               const double*                                               src,
                               double*                                                     dst) const;

    // The CUDAWrappers::MatrixFree object doesn't know about the number
    // of degrees of freedom and the number of quadrature points so we need
    // to store these for index calculations in the call operator.
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = dealii::Utilities::pow(n_dofs_1d, dim);
    static const unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);
  };


  // This is the call operator that performs the Advection operator evaluation
  // on a given cell similar to the MatrixFree framework on the CPU.
  // In particular, we need access values of the source vector and we write
  // value information to the destination vector.
  //
  template <int dim, int fe_degree, int n_q_points_1d>
  __device__ void LocalAdvectionOperator<dim, fe_degree, n_q_points_1d>::
  operator()(const unsigned int                                          cell,
             const typename CUDAWrappers::MatrixFree<dim, double>::Data* gpu_data,
             CUDAWrappers::SharedData<dim, double>*                      shared_data,
             const double*                                               src,
             double*                                                     dst) const {
    CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, double> fe_eval(cell, gpu_data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, false); /*--- This two-stage procedure is necessary because gather_evaluate has not been defined ---*/

    fe_eval.apply_for_each_quad_point(AdvectionOperatorQuad<dim, fe_degree>());

    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(dst); /*--- This two-stage procedure is necessary because integrate_scatter has not been defined ---*/
  }


  // @sect{Class <code>AdvectionOperator</code>}

  // The `AdvectionOperator` class acts as wrapper for
  // `LocalAdvectionOperator` defining an interface that can be used
  // with linear solvers like SolverCG. In particular, like every
  // class that implements the interface of a linear operator, it
  // needs to have a `vmult()` function that performs the action of
  // the linear operator on a source vector.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping = 1>
  class AdvectionOperator {
  public:
    AdvectionOperator(const DoFHandler<dim>&           dof_handler,
                      const AffineConstraints<double>& constraints);

    void initialize_dof_vector(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& vec) const;

    void vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>&       dst,
               const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& src) const;

  private:
    CUDAWrappers::MatrixFree<dim, double> mf_data; /*--- Notice that this class cannot be derived from MatrixFreeOperators::Base
                                                         because we need CUDAWrappers::MatrixFree instance ---*/
  };


  // The following is the implementation of the constructor of this
  // class. In the first part, We initialize the `mf_data` member
  // variable that is going to provide us with the necessary
  // information when evaluating the operator.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  AdvectionOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  AdvectionOperator(const DoFHandler<dim>&           dof_handler,
                    const AffineConstraints<double>& constraints) {
    MappingQGeneric<dim> mapping(degree_mapping); /*--- Mapping ---*/

    typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData additional_data; /*--- Additional data with flags to be initialized ---*/
    additional_data.mapping_update_flags = update_values | update_JxW_values | update_quadrature_points;

    const QGauss<1> quad(n_q_points_1d); /*--- Quadrature formula ---*/

    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data); /*--- Reinit the matrix free structure ---*/
  }


  // Auxiliary function to initialize vectors since we cannot derive from MatrixFreeOperators and, therefore,
  // we cannot use a MatrixFree to directly initialize the AdvectionOperator and the corresponding vectors.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  initialize_dof_vector(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& vec) const {
    mf_data.initialize_dof_vector(vec);
  }


  // The key step then is to use all of the previous classes to loop over
  // all cells to perform the matrix-vector product. We implement this
  // in the next function.
  //
  // When applying the LocalAdvectionOperator, we have to be careful to handle
  // boundary conditions correctly. Since the local operator doesn't know about
  // constraints, we have to copy the correct values from the source to the
  // destination vector afterwards.
  //
  template <int dim, int fe_degree, int n_q_points_1d, int degree_mapping>
  void AdvectionOperator<dim, fe_degree, n_q_points_1d, degree_mapping>::
  vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>&       dst,
        const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>& src) const {
    dst = 0;

    LocalAdvectionOperator<dim, fe_degree, n_q_points_1d> local_advection_operator;
    mf_data.cell_loop(local_advection_operator, src, dst);
    /*--- The cell_loop needs as first input argument a Functor with a __device__ void operator().
          Hence, we need a class for the local action and a simple routine seems not to be sufficient because of the operator() request. ---*/

    mf_data.copy_constrained_values(src, dst);
  }


  // @sect{Class <code>AdvectionProblem</code>}

  // This is the main class of this program. It defines the usual
  // framework we use for tutorial programs. The only point worth
  // commenting on is the `solve()` function and the choice of vector
  // types.
  //
  template <int dim, int fe_degree>
  class AdvectionProblem {
  public:
    AdvectionProblem(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose, const unsigned int output_interval);

  protected:
    const double t_0; /*--- Initial time auxiliary variable ----*/
    const double T;   /*--- Final time auxiliary variable ----*/
    double       dt;  /*--- Time step auxiliary variable ---*/

    parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

    FE_Q<dim>       fe;   /*--- Finite element space ---*/

    DoFHandler<dim> dof_handler; /*--- Degrees of freedom handler ---*/

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
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host_old;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host_tmp;

  private:
    EquationData::Density<dim>  rho_init;
    EquationData::Velocity<dim> velocity;

    std::unique_ptr<AdvectionOperator<dim, fe_degree, fe_degree + 1>> system_matrix_dev;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_refines);

    void setup_system();

    void initialize();

    void assemble_rhs();

    void solve();

    void output_results(const unsigned int step);

    void analyze_results();

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
    double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

    std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

    /*--- Now we declare a bunch of variables for text output ---*/
    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_n_dofs_density,
                  output_error_rho;
  };


  // The implementation of all the remaining functions of this class apart from
  // `Advectionproblem::solve()` doesn't contain anything new and we won't
  // further comment much on the overall approach.
  //
  template <int dim, int fe_degree>
  AdvectionProblem<dim, fe_degree>::AdvectionProblem(RunTimeParameters::Data_Storage& data) :
    t_0(data.initial_time),
    T(data.final_time),
    dt(data.dt),
    triangulation(MPI_COMM_WORLD),
    fe(fe_degree),
    dof_handler(triangulation),
    rho_init(data.initial_time),
    velocity(data.initial_time),
    max_its(data.max_iterations),
    eps(data.eps),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out),
    output_error_rho("./" + data.dir + "/error_analysis_rho.dat", std::ofstream::out) {
      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      create_triangulation(data.n_global_refines);
      setup_system();
      initialize();
    }


  // Build the domain
  //
  template<int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::create_triangulation(const unsigned int n_refines) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, true);

    triangulation.refine_global(n_refines);
  }


  // Setup the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::setup_system() {
    TimerOutput::Scope t(time_table, "Setup system");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
    DoFTools::make_periodicity_constraints(dof_handler, 2, 3, 1, constraints);
    constraints.close();

    solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_host_old.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_host_tmp.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    system_matrix_dev.reset(new AdvectionOperator<dim, fe_degree, fe_degree + 1>(dof_handler, constraints));
    system_matrix_dev->initialize_dof_vector(solution_dev);
    system_matrix_dev->initialize_dof_vector(system_rhs_dev);
  }


  // Initialize the field
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler, rho_init, solution_host);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(solution_host, VectorOperation::insert);
    solution_dev.import(rw_vector, VectorOperation::insert);
  }


  // Unlike programs such as step-4 or step-6, we will not have to
  // assemble the whole linear system but only the right hand side
  // vector.
  //
  // At the end of the function, we can't directly copy the values
  // from the host to the device but need to use an intermediate
  // object of type LinearAlgebra::ReadWriteVector to construct the
  // correct communication pattern necessary.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_rhs() {
    TimerOutput::Scope t(time_table, "Assemble rhs");

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(locally_owned_dofs,
                                                                                  locally_relevant_dofs,
                                                                                  MPI_COMM_WORLD); /*--- Right hand-side vector ---*/

    const QGauss<dim> quadrature_formula(fe_degree + 1); /*---Quadrature formula ---*/

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> old_solution_values(n_q_points);
    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        cell_rhs = 0;

        fe_values.reinit(cell);

        fe_values.get_function_values(solution_host, old_solution_values);
        fe_values.get_function_gradients(solution_host, old_solution_gradients);

        for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
          const auto& x_q = fe_values.quadrature_point(q_index);
          Tensor<1, dim> u;
          for(unsigned int d = 0; d < dim; ++d) {
            u[d] = velocity.value(x_q, d);
          }

          for(unsigned int i = 0; i < dofs_per_cell; ++i) {
            cell_rhs(i) += fe_values.shape_value(i, q_index)*
                           (old_solution_values[q_index] - dt*scalar_product(u, old_solution_gradients[q_index]))*
                           fe_values.JxW(q_index);
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
  void AdvectionProblem<dim, fe_degree>::solve() {
    TimerOutput::Scope t(time_table, "Update density");

    PreconditionIdentity preconditioner;

    SolverControl solver_control(max_its, eps*system_rhs_dev.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>> cg(solver_control);
    cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(solution_dev, VectorOperation::insert);
    solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(solution_host);
  }

  // The output results function is as usual since we have already copied the
  // values back from the GPU to the CPU.
  //
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::output_results(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output results");

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    solution_host.update_ghost_values();
    data_out.add_data_vector(solution_host, "solution");

    data_out.build_patches();

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    std::string output_file = "./" + saving_dir + "/solution_" + Utilities::int_to_string(step, 5) + ".vtu";;
    data_out.write_vtu_in_parallel(output_file, MPI_COMM_WORLD);
  }


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::analyze_results() {
    TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

    QGauss<dim> quadrature_formula(EquationData::degree + 1);

    Vector<double> L2_error_per_cell_rho;
    L2_error_per_cell_rho.reinit(triangulation.n_active_cells());

    VectorTools::integrate_difference(dof_handler, solution_host, rho_init,
                                      L2_error_per_cell_rho, quadrature_formula, VectorTools::L2_norm);
    const double error_L2_rho = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);

    solution_host_tmp = 0;
    VectorTools::integrate_difference(dof_handler, solution_host_tmp, rho_init,
                                      L2_error_per_cell_rho, quadrature_formula, VectorTools::L2_norm);
    const double L2_rho = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    const double error_rel_L2_rho = error_L2_rho/L2_rho;

    /*--- Save results ---*/
    pcout << "Verification via L2 error:    "          << error_L2_rho     << std::endl;
    pcout << "Verification via L2 relative error:    " << error_rel_L2_rho << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_error_rho << error_L2_rho     << std::endl;
      output_error_rho << error_rel_L2_rho << std::endl;
    }
  }


  // There is nothing surprising in the `run()` function either. We simply
  // call all the previous implemented functions.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    analyze_results();
    output_results(0);

    double time    = t_0;
    unsigned int n = 0;

    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      solution_host_old.equ(1.0, solution_host);

      assemble_rhs();
      solve();

      assemble_rhs();
      solve();
      solution_host *= 0.25;
      solution_host.add(0.75, solution_host_old);

      assemble_rhs();
      solve();
      solution_host *= 2.0/3.0;
      solution_host.add(1.0/3.0, solution_host_old);

      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
      }
    }
    analyze_results();
    /*--- Save the final results if not previously done ---*/
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
  }

} // namespace AdvectionProblem


// @sect3{The <code>main()</code> function}

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
    using namespace AdvectionSolver;

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

    AdvectionProblem<2, EquationData::degree> advection_problem(data);
    advection_problem.run(data.verbose, data.output_interval);
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
