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
#include <deal.II/numerics/matrix_tools.h>

// The following ones include the data structures for the
// implementation of matrix-based methods on GPU:
#include <deal.II/base/cuda.h>

#include <deal.II/lac/cuda_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

// Include some other fundamental headers
#include <fstream>

#include "runtime_parameters.h"
#include "equation_data.h"


// As usual, we enclose everything into a namespace of its own:
//
namespace AdvectionSolver {
  using namespace dealii;

  // @sect{Class <code>AdvectionProblem</code>}

  template <int dim, int fe_degree>
  class AdvectionProblem {
  public:
    AdvectionProblem(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose, const unsigned int output_interval);

  protected:
    const double t_0; /*--- Initial time auxiliary variable ----*/
    const double T;   /*--- Final time auxiliary variable ----*/
    double       dt;  /*--- Time step auxiliary variable ---*/

    Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

    FE_Q<dim>       fe; /*--- Finite element space ---*/

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
    CUDAWrappers::SparseMatrix<double> system_matrix_dev;
    SparsityPattern sparsity_pattern;

    LinearAlgebra::CUDAWrappers::Vector<double> solution_dev;
    LinearAlgebra::CUDAWrappers::Vector<double> system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host_old;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host_tmp;

  private:
    EquationData::Density<dim>  rho_init;
    EquationData::Velocity<dim> velocity;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_refines);

    void setup_system();

    void initialize();

    void assemble_matrix();

    void assemble_rhs();

    void solve();

    void output_results(const unsigned int step);

    void analyze_results();

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

    /*--- CUDA related variable ---*/
    Utilities::CUDA::Handle cuda_handle;
  };


  // Class constructor
  //
  template <int dim, int fe_degree>
  AdvectionProblem<dim, fe_degree>::AdvectionProblem(RunTimeParameters::Data_Storage& data) :
    t_0(data.initial_time),
    T(data.final_time),
    dt(data.dt),
    triangulation(),
    fe(fe_degree),
    dof_handler(triangulation),
    rho_init(data.initial_time),
    velocity(data.initial_time),
    max_its(data.max_iterations),
    eps(data.eps),
    saving_dir(data.dir),
    pcout(std::cout),
    time_out("./" + data.dir + "/time_analysis_1proc.dat"),
    ptime_out(time_out),
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

    GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, false);

    triangulation.refine_global(n_refines);
  }


  // Setup of the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::setup_system() {
    TimerOutput::Scope t(time_table, "Setup system");

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    solution_dev.reinit(dof_handler.n_dofs());
    system_rhs_dev.reinit(dof_handler.n_dofs());

    solution_host.reinit(dof_handler.n_dofs());
    solution_host_old.reinit(dof_handler.n_dofs());
    solution_host_tmp.reinit(dof_handler.n_dofs());
  }


  // Initialize the field
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler, rho_init, solution_host);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(solution_host, VectorOperation::insert);
    solution_dev.import(rw_vector, VectorOperation::insert);
  }


  // We assemble now the matrix
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_matrix() {
    SparseMatrix<double> system_matrix_host;
    system_matrix_host.reinit(sparsity_pattern);

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      cell_matrix = 0;

      fe_values.reinit(cell);

      for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            cell_matrix(i,j) += fe_values.shape_value(i, q_index)*
                                (fe_values.shape_value(j, q_index))*
                                fe_values.JxW(q_index);
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices,
                                             system_matrix_host);
    }
    system_matrix_host.compress(VectorOperation::add);

    system_matrix_dev.reinit(cuda_handle, system_matrix_host);
  }


  // We assemble now the rhs
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_rhs() {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(dof_handler.n_dofs()); /*--- Right hand-side vector ---*/

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> old_solution_values(n_q_points);
    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
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
    system_rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import(rw_vector, VectorOperation::insert);
  }


  // Solve the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::solve() {
    TimerOutput::Scope t(time_table, "Update density");

    PreconditionIdentity preconditioner;

    SolverControl solver_control(max_its, eps*system_rhs_dev.l2_norm());
    SolverCG<LinearAlgebra::CUDAWrappers::Vector<double>> cg(solver_control);
    cg.solve(system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(solution_dev, VectorOperation::insert);
    solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(solution_host);
  }


  // The output results function is as usual since we have already copied the
  // values back from the GPU to the CPU.
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
    std::ofstream output_file("./" + saving_dir + "/solution_" + Utilities::int_to_string(step, 5) + ".vtu");
    data_out.write_vtu(output_file);

    solution_host.zero_out_ghost_values();
  }


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::analyze_results() {
    TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

    QGauss<dim> quadrature_formula(fe_degree + 1);

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

    output_error_rho << error_L2_rho     << std::endl;
    output_error_rho << error_rel_L2_rho << std::endl;
  }


  // There is nothing surprising in the `run()` function either. We simply
  // compute the solution on a series of (globally) refined meshes.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose);

    analyze_results();
    output_results(0);

    double time    = t_0;
    unsigned int n = 0;

    assemble_matrix();

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
} // namespace AdvectionSolver


// @sect3{The <code>main()</code> function}

// Finally for the `main()` function.  By default, all the MPI ranks
// will try to access the device with number 0, which we assume to be
// the GPU device associated with the CPU.
//
int main() {
  try
  {
    using namespace AdvectionSolver;

    cudaError_t cuda_error_code = cudaSetDevice(0);
    AssertCuda(cuda_error_code);

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    deallog.depth_console(data.verbose == true? 2 : 0);

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
