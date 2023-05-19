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
#include <deal.II/grid/grid_refinement.h>
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
  const double t_0;              /*--- Initial time auxiliary variable ----*/
  const double T;                /*--- Final time auxiliary variable ----*/
  unsigned int HYPERBOLIC_stage; /*--- Flag to check at which current stage of the IMEX we are ---*/
  double       dt;               /*--- Time step auxiliary variable ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  /*--- Finite element spaces for all the variables ---*/
  FESystem<dim> fe_density;
  FESystem<dim> fe_velocity;

  /*--- Degrees of freedom handlers for all the variables ---*/
  DoFHandler<dim> dof_handler_density;
  DoFHandler<dim> dof_handler_velocity;

  /*--- Auxiliary quadratures for all the variables ---*/
  QGaussLobatto<dim> quadrature_density;
  QGaussLobatto<dim> quadrature_velocity;

  /*--- Variables for the density ---*/
  LinearAlgebra::distributed::Vector<double> rho_old;
  LinearAlgebra::distributed::Vector<double> rho_tmp_2;
  LinearAlgebra::distributed::Vector<double> rho_tmp_3;
  LinearAlgebra::distributed::Vector<double> rho_curr;
  LinearAlgebra::distributed::Vector<double> rhs_rho;

  /*--- Variables for the velocity ---*/
  LinearAlgebra::distributed::Vector<double> u;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation(const unsigned int n_refines); /*--- Function to create the grid ---*/

  void setup_dofs(); /*--- Function to set the dofs ---*/

  void initialize(); /*--- Function to initialize the fields ---*/

  void update_density(); /*--- Function to update the density ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

private:
  EquationData::Velocity<dim> u_init;
  EquationData::Density<dim>  rho_init;

  /*--- Auxiliary structures for the matrix-free and for the multigrid ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  AdvectionOperator<dim, EquationData::degree, 2*EquationData::degree + 1,
                    LinearAlgebra::distributed::Vector<double>> advection_matrix;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<double>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<double> constraints_velocity,
                            constraints_density;

  std::vector<QGauss<1>> quadratures;  /*--- Auxiliary container for the quadrature in matrix-free ---*/

  unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

  std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  /*--- Now we declare a bunch of variables for text output ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_n_dofs_density;

  double get_maximal_velocity(); /*--- Get maximal velocity to compute the Courant number ---*/
};


// @sect{ <code>AdvectionSolver::AdvectionSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
AdvectionSolver<dim>::AdvectionSolver(RunTimeParameters::Data_Storage& data):
  t_0(data.initial_time),
  T(data.final_time),
  HYPERBOLIC_stage(1),            //--- Initialize the flag for the time advancing scheme
  dt(data.dt),
  triangulation(MPI_COMM_WORLD, parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_density(FE_Q<dim>(EquationData::degree), 1),
  fe_velocity(FE_Q<dim>(EquationData::degree), dim),
  dof_handler_density(triangulation),
  dof_handler_velocity(triangulation),
  quadrature_density(EquationData::degree + 1),
  quadrature_velocity(EquationData::degree + 1),
  u_init(data.initial_time),
  rho_init(data.initial_time),
  advection_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    dof_handlers.clear();

    constraints.clear();
    constraints_velocity.clear();
    constraints_density.clear();

    quadratures.clear();

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

  GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, true);

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
  dof_handler_velocity.distribute_dofs(fe_velocity);
  dof_handler_density.distribute_dofs(fe_density);

  pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
        << std::endl
        << "dim (X_h) = " << dof_handler_density.n_dofs() << std::endl
        << std::endl;

  /*--- Save the number of degrees of freedom just for info ---*/
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_n_dofs_density << dof_handler_density.n_dofs() << std::endl;
  }

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_gradients | update_JxW_values |
                                                         update_quadrature_points | update_values);
  additional_data.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors | update_values);
  additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors | update_values);
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  /*--- Set the container with the dof handlers ---*/
  dof_handlers.push_back(&dof_handler_velocity);
  dof_handlers.push_back(&dof_handler_density);

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints.push_back(&constraints_velocity);
  DoFTools::make_hanging_node_constraints(dof_handler_density, constraints_density);
  DoFTools::make_periodicity_constraints(dof_handler_density, 0, 1, 0, constraints_density);
  DoFTools::make_periodicity_constraints(dof_handler_density, 2, 3, 1, constraints_density);
  constraints_density.close();
  constraints.push_back(&constraints_density);

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(2*EquationData::degree + 1));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(MappingQ1<dim>(), dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the velocity ---*/
  matrix_free_storage->initialize_dof_vector(u, 0);

  /*--- Initialize the variables related to the density ---*/
  matrix_free_storage->initialize_dof_vector(rho_old, 1);
  matrix_free_storage->initialize_dof_vector(rho_tmp_2, 1);
  matrix_free_storage->initialize_dof_vector(rho_tmp_3, 1);
  matrix_free_storage->initialize_dof_vector(rho_curr, 1);
  matrix_free_storage->initialize_dof_vector(rhs_rho, 1);
}


// @sect{ <code>AdvectionSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void AdvectionSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  VectorTools::interpolate(dof_handler_density, rho_init, rho_old);
  VectorTools::interpolate(dof_handler_velocity, u_init, u);
}


// @sect{<code>AdvectionSolver::update_density</code>}

// This implements the update of the density for the hyperbolic part
//
template<int dim>
void AdvectionSolver<dim>::update_density() {
  TimerOutput::Scope t(time_table, "Update density");

  const std::vector<unsigned int> tmp = {1};

  advection_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(HYPERBOLIC_stage == 1) {
    advection_matrix.vmult_rhs_update(rhs_rho, {rho_old});
  }
  else if(HYPERBOLIC_stage == 2) {
    advection_matrix.vmult_rhs_update(rhs_rho, {rho_tmp_2});
  }
  else {
    advection_matrix.vmult_rhs_update(rhs_rho, {rho_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_rho.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  if(HYPERBOLIC_stage == 1) {
    rho_tmp_2.equ(1.0, rho_old);
    cg.solve(advection_matrix, rho_tmp_2, rhs_rho, PreconditionIdentity());

    constraints_density.distribute(rho_tmp_2);
  }
  else if(HYPERBOLIC_stage == 2) {
    rho_tmp_3.equ(1.0, rho_tmp_2);
    cg.solve(advection_matrix, rho_tmp_3, rhs_rho, PreconditionIdentity());

    rho_tmp_3 *= 0.25;
    rho_tmp_3.add(0.75, rho_old);

    constraints_density.distribute(rho_tmp_3);
  }
  else {
    rho_curr.equ(1.0, rho_tmp_3);
    cg.solve(advection_matrix, rho_curr, rhs_rho, PreconditionIdentity());

    rho_curr *= 2.0/3.0;
    rho_curr.add(1.0/3.0, rho_old);

    constraints_density.distribute(rho_curr);
  }
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

  rho_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});

  std::vector<std::string> velocity_names(dim, "u");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  u.update_ghost_values();

  data_out.build_patches();
  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
}


// The following function is used in determining the maximal velocity
// in order to compute the CFL
//
template<int dim>
double AdvectionSolver<dim>::get_maximal_velocity() {
  return u.linfty_norm();
}


// @sect{ <code>AdvectionSolver::run</code> }

// This is the time marching function, which starting at <code>t_0</code>
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

  output_results(0);
  double time = t_0;
  unsigned int n = 0;
  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- First stage of the IMEX operator ---*/
    HYPERBOLIC_stage = 1;

    verbose_cout << "  Update stage 1" << std::endl;
    update_density();

    /*--- Second stage of IMEX operator ---*/
    HYPERBOLIC_stage = 2;

    verbose_cout << "  Update stage 2" << std::endl;
    update_density();

    /*--- Final stage of RK scheme to update ---*/
    HYPERBOLIC_stage = 3;

    verbose_cout << "  Update stage 3" << std::endl;
    update_density();

    /*--- Update density and velocity before applying the damping layers ---*/
    rho_old.equ(1.0, rho_curr);

    const double max_velocity = get_maximal_velocity();
    pcout<< "Maximal velocity = " << max_velocity << std::endl;
    pcout << "CFL_u = " << dt*max_velocity*EquationData::degree*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
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
int main(int argc, char *argv[]) {
  try {
    using namespace Advection;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

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
