/* Author: Giuseppe Orlando, 2023. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include "advection_operator.h"

using namespace Advection;

// @sect{The <code>AdvectionSolver</code> class}

// Now for the main class of the program. It implements the solver for the
// Euler equations using the discretization previously implemented.
//
template<int dim>
class AdvectionSolver {
public:
  AdvectionSolver(RunTimeParameters::Data_Storage& data); /*--- Class constructor ---*/

  void run(const bool verbose = false, const unsigned int output_interval = 10);
  /*--- The run function which actually runs the simulation ---*/

protected:
  const double t0;        /*--- Initial time auxiliary variable ----*/
  const double T;         /*--- Final time auxiliary variable ----*/
  double       dt;        /*--- Time step auxiliary variable ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  FE_Q<dim> fe; /*--- Finite element space ---*/

  DoFHandler<dim> dof_handler; /*--- Degrees of freedom handler ---*/

  MappingQ1<dim>  mapping; /*--- Employed mapping for the sake of generality ---*/

  /*--- Variables for the solution ---*/
  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> solution_old;
  LinearAlgebra::distributed::Vector<double> solution_tmp;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation(const unsigned int n_refines); /*--- Function to create the grid ---*/

  void setup_dofs(); /*--- Function to set the dofs ---*/

  void initialize(); /*--- Function to initialize the fields ---*/

  void update_solution(); /*--- Function to update the solution ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

  void analyze_results(); /*--- Compute errors ---*/

private:
  EquationData::Density<dim> solution_init;

  /*--- Auxiliary structures for the matrix-free and for the multigrid ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  AdvectionOperator<dim, EquationData::degree, EquationData::degree + 1,
                    LinearAlgebra::distributed::Vector<double>> advection_matrix;

  AffineConstraints<double> constraints;

  unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

  std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  /*--- Now we declare a bunch of variables for text output ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_n_dofs,
                output_error;
};


// @sect{ <code>AdvectionSolver::AdvectionSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
AdvectionSolver<dim>::AdvectionSolver(RunTimeParameters::Data_Storage& data):
  t0(data.initial_time),
  T(data.final_time),
  dt(data.dt),
  triangulation(MPI_COMM_WORLD),
  fe(EquationData::degree),
  dof_handler(triangulation),
  mapping(),
  solution_init(data.initial_time),
  advection_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_n_dofs("./" + data.dir + "/n_dofs.dat", std::ofstream::out),
  output_error("./" + data.dir + "/error_analysis.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    create_triangulation(data.n_global_refines);
    setup_dofs();
    initialize();
  }


// @sect{<code>AdvectionSolver::create_triangulation_and_dofs</code>}

// The method that creates the triangulation.
//
template<int dim>
void AdvectionSolver<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, false);

  triangulation.refine_global(n_refines);
}


// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom and renumbers them, and
// initializes the matrices and vectors that we will use.
//
template<int dim>
void AdvectionSolver<dim>::setup_dofs() {
  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Set degrees of freedom ---*/
  dof_handler.distribute_dofs(fe);

  pcout << "dim (X_h) = " << dof_handler.n_dofs() << std::endl
        << std::endl;

  /*--- Save the number of degrees of freedom just for info ---*/
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_n_dofs << dof_handler.n_dofs() << std::endl;
  }

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients | update_JxW_values | update_quadrature_points;

  /*--- Set the the constraints ---*/
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(mapping, dof_handler, constraints, QGauss<1>(EquationData::degree + 1), additional_data);
  advection_matrix.initialize(matrix_free_storage);

  /*--- Initialize the variables related to the solution ---*/
  matrix_free_storage->initialize_dof_vector(solution);
  matrix_free_storage->initialize_dof_vector(solution_old);
  matrix_free_storage->initialize_dof_vector(solution_tmp);
  matrix_free_storage->initialize_dof_vector(system_rhs);
}


// @sect{ <code>AdvectionSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void AdvectionSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  VectorTools::interpolate(mapping, dof_handler, solution_init, solution);
}


// @sect{<code>AdvectionSolver::update_solution</code>}

// This implements the update of the solution
//
template<int dim>
void AdvectionSolver<dim>::update_solution() {
  TimerOutput::Scope t(time_table, "Update solution");

  advection_matrix.vmult_rhs_update(system_rhs, solution);
  solution.zero_out_ghost_values();

  SolverControl solver_control(max_its, eps*system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  cg.solve(advection_matrix, solution, system_rhs, PreconditionIdentity());

  constraints.distribute(solution);
}


// @sect{ <code>AdvectionSolver::output_results</code> }

// This method plots the current solution. The main difficulty is that we want
// to create a single output file that contains the data for all velocity
// components and the pressure. On the other hand, velocities and the pressure
// live on separate DoFHandler objects, and
// so can't be written to the same file using a single DataOut object. As a
// consequence, we have to work a bit harder to get the various pieces of data
// into a single DoFHandler object, and then use that to drive graphical
// output.
//
template<int dim>
void AdvectionSolver<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  DataOut<dim> data_out;

  solution.update_ghost_values();

  data_out.add_data_vector(dof_handler, solution, "solution", {DataComponentInterpretation::component_is_scalar});
  data_out.build_patches(mapping, EquationData::degree);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  solution.zero_out_ghost_values();
}

// @sect{ <code>AdvectionSolver::analyze_results</code> }

// Since we have solved a problem with analytic solution, we want to verify
// the correctness of our implementation by computing the errors of the
// numerical result against the analytic solution.
//
template <int dim>
void AdvectionSolver<dim>::analyze_results() {
  TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

  QGauss<dim> quadrature_formula(EquationData::degree + 1);

  Vector<double> L2_error_per_cell;
  L2_error_per_cell.reinit(triangulation.n_active_cells());

  solution.update_ghost_values();
  VectorTools::integrate_difference(mapping, dof_handler, solution, solution_init,
                                    L2_error_per_cell, quadrature_formula, VectorTools::L2_norm);
  solution.zero_out_ghost_values();
  const double error_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell, VectorTools::L2_norm);

  solution_tmp = 0;
  solution_tmp.update_ghost_values();
  VectorTools::integrate_difference(mapping, dof_handler, solution_tmp, solution_init,
                                    L2_error_per_cell, quadrature_formula, VectorTools::L2_norm);
  const double L2_rho = VectorTools::compute_global_error(triangulation, L2_error_per_cell, VectorTools::L2_norm);
  const double error_rel_L2 = error_L2/L2_rho;

  /*--- Save results ---*/
  pcout << "Verification via L2 error:    "          << error_L2     << std::endl;
  pcout << "Verification via L2 relative error:    " << error_rel_L2 << std::endl;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_error << error_L2     << std::endl;
    output_error << error_rel_L2 << std::endl;
  }
}


// @sect{ <code>AdvectionSolver::run</code> }

// This is the time marching function, which starting at <code>t0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<int dim>
void AdvectionSolver<dim>::run(const bool verbose, const unsigned int output_interval) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  analyze_results();
  output_results(0);

  double time    = t0;
  unsigned int n = 0;

  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    solution_old.equ(1.0, solution);

    /*--- First stage of the SSP scheme ---*/
    verbose_cout << "  Update stage 1" << std::endl;
    update_solution();

    /*--- Second stage of SSP scheme ---*/
    verbose_cout << "  Update stage 2" << std::endl;
    update_solution();
    solution *= 0.25;
    solution.add(0.75, solution_old);

    /*--- Final stage of SSP scheme ---*/
    verbose_cout << "  Update stage 3" << std::endl;
    update_solution();
    solution *= 2.0/3.0;
    solution.add(1.0/3.0, solution_old);

    /*--- Save the results each 'output_interval' steps ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
    if(T - time < dt && T - time > 1e-10) {
      /*--- Recompute and rest the time if needed towards the end of the simulation to stop at the proper final time ---*/
      dt = T - time;
      advection_matrix.set_dt(dt);
    }
  }
  analyze_results();
  /*--- Save the final results if not previously done ---*/
  if(n % output_interval != 0) {
    verbose_cout << "Plotting Solution final" << std::endl;
    output_results(n);
  }
}


// @sect{ The main function }

// The main function is quite standard. We just need to declare the AdvectionSolver
// instance and let the simulation run.
//
int main(int argc, char* argv[]) {
  try {
    using namespace Advection;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    AdvectionSolver<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception& exc) {
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
  catch(...) {
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

}
