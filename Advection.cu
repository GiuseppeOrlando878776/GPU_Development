/*--- Author: Giuseppe Orlando, 2023 ---*/

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/cuda.h>

#include <deal.II/lac/cuda_sparse_matrix.h>
#include <deal.II/lac/cuda_precondition.h>

#include <fstream>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/copy_data.h>

#include <deal.II/fe/fe_interface_values.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// As usual, we enclose everything into a namespace of its own:
//
namespace AdvectionSolver {
  using namespace dealii;

  /*--- Auxiliry structs to coyp data from local to global in a DG framework ---*/
  struct CopyDataFace {
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> joint_dof_indices;
  };


  struct CopyData {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template<class Iterator>
    void reinit(const Iterator& cell, unsigned int dofs_per_cell) {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };


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

    parallel::distributed::Triangulation<dim> triangulation;

    FE_DGQ<dim> fe;

    DoFHandler<dim> dof_handler;

    // Since all the operations in the `solve()` function are executed on the
    // graphics card, it is necessary for the vectors used to store their values
    // on the GPU as well. In addition, we also keep a solution vector with
    // CPU storage such that we can view and display the solution as usual.
    CUDAWrappers::SparseMatrix<double> system_matrix_dev;
    SparsityPattern sparsity_pattern;

    LinearAlgebra::CUDAWrappers::Vector<double> solution_dev;
    LinearAlgebra::CUDAWrappers::Vector<double> system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host,
                                                                  solution_host_old,
                                                                  solution_host_tmp;

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

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    unsigned int n_refines; /*--- Number of refinements auxiliary variable ---*/

    unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
    double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

    std::string  saving_dir;      /*--- Auxiliary variable for the directory to save the results ---*/

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
    triangulation(MPI_COMM_WORLD),
    fe(fe_degree),
    dof_handler(triangulation),
    rho_init(data.initial_time),
    velocity(data.initial_time),
    n_refines(data.n_global_refines),
    max_its(data.max_iterations),
    eps(data.eps),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_1proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out),
    output_error_rho("./" + data.dir + "/error_analysis_rho.dat", std::ofstream::out) {
      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      create_triangulation(n_refines);
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


  // Setup of the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::setup_system() {
    TimerOutput::Scope t(time_table, "Setup system");

    dof_handler.distribute_dofs(fe);

    pcout << "dim (Q_h) = " << dof_handler.n_dofs() << std::endl
          << std::endl;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_n_dofs_density << dof_handler.n_dofs() << std::endl;
    }

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
    DoFTools::make_periodicity_constraints(dof_handler, 2, 3, 1, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    solution_dev.reinit(dof_handler.n_dofs());
    system_rhs_dev.reinit(dof_handler.n_dofs());

    solution_host.reinit(dof_handler.n_dofs());
    solution_host_old.reinit(solution_host);
    solution_host_tmp.reinit(solution_host);
  }


  // Initialize the field
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(MappingQ1<dim>(), dof_handler, rho_init, solution_host);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(solution_host, VectorOperation::insert);
    solution_dev.import(rw_vector, VectorOperation::insert);
  }


  // We assemble now the matrix
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_matrix() {
    TimerOutput::Scope t(time_table, "Assemble matrix");

    SparseMatrix<double> system_matrix_host;
    system_matrix_host.reinit(sparsity_pattern);

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    /*--- Compute cellwise contribution ---*/
    const auto cell_worker = [&](const Iterator&                    cell,
                                 MeshWorker::ScratchData<dim, dim>& scratch_data,
                                 CopyData&                          copy_data) {
      const FEValues<dim>& fe_values = scratch_data.reinit(cell);

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, dofs_per_cell);

      for(unsigned int q_index = 0; q_index < fe_values.n_quadrature_points; ++q_index) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            copy_data.cell_matrix(i, j) += fe_values.shape_value(i, q_index)*
                                           (fe_values.shape_value(j, q_index))*
                                           fe_values.JxW(q_index);
          }
        }
      }
    };

    /*--- Auxiliary lambda function to copy data from local to global ---*/
    const auto copier = [&](const CopyData& c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.local_dof_indices,
                                             system_matrix_host);
    };

    /*--- Create the needed auxiliary structues (i.e. quadrature formula, scratch data and copy data) ---*/
    const QGauss<dim> quadrature_formula_cell(fe_degree + 1);
    MeshWorker::ScratchData<dim, dim> scratch_data(fe,
                                                   quadrature_formula_cell,
                                                   update_values | update_quadrature_points | update_JxW_values);
    CopyData copy_data;

    /*--- Perform the loop that effectively builds the matrix ---*/
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

    system_matrix_dev.reinit(cuda_handle, system_matrix_host);
  }


  // We assemble now the rhs
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_rhs() {
    TimerOutput::Scope t(time_table, "Assemble rhs");

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(locally_owned_dofs,
                                                                                  locally_relevant_dofs,
                                                                                  MPI_COMM_WORLD); /*--- Right hand-side vector ---*/

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    /*--- Compute cellwise contribution ---*/
    const auto cell_worker = [&](const Iterator&                    cell,
                                 MeshWorker::ScratchData<dim, dim>& scratch_data,
                                 CopyData&                          copy_data) {
      const FEValues<dim>& fe_values = scratch_data.reinit(cell);

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, dofs_per_cell);

      const unsigned int n_q_points = fe_values.n_quadrature_points;
      std::vector<double> old_solution_values(n_q_points);
      fe_values.get_function_values(solution_host, old_solution_values);

      for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        const auto& x_q = fe_values.quadrature_point(q_index);
        Tensor<1, dim> u;
        for(unsigned int d = 0; d < dim; ++d) {
          u[d] = velocity.value(x_q, d);
        }

        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          copy_data.cell_rhs(i) += (fe_values.shape_value(i, q_index)*old_solution_values[q_index] +
                                    dt*old_solution_values[q_index]*scalar_product(u, fe_values.shape_grad(i, q_index)))*
                                   fe_values.JxW(q_index);
        }
      }
    };

    /*--- Compute single inner face contribution ---*/
    const auto face_worker = [&](const Iterator&                    cell,
                                 const unsigned int&                f,
                                 const unsigned int&                sf,
                                 const Iterator&                    ncell,
                                 const unsigned int&                nf,
                                 const unsigned int&                nsf,
                                 MeshWorker::ScratchData<dim, dim>& scratch_data,
                                 CopyData&                          copy_data) {
      const FEInterfaceValues<dim>& fe_iv = scratch_data.reinit(cell, f, sf, ncell, nf, nsf);

      copy_data.face_data.emplace_back();
      CopyDataFace& copy_data_face     = copy_data.face_data.back();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

      const unsigned int n_q_points = fe_iv.n_quadrature_points;
      std::vector<double> old_solution_average_values(n_q_points);
      std::vector<double> old_solution_jump_values(n_q_points);
      fe_iv.get_average_of_function_values(solution_host, old_solution_average_values);
      fe_iv.get_jump_in_function_values(solution_host, old_solution_jump_values);

      const auto& quadrature_points = fe_iv.get_quadrature_points();
      const auto& normal_vectors    = fe_iv.get_normal_vectors();

      const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
      copy_data_face.cell_rhs.reinit(n_dofs);

      for(unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        const auto& n   = normal_vectors[q_index];

        const auto& x_q = quadrature_points[q_index];
        Tensor<1, dim> u;
        for(unsigned int d = 0; d < dim; ++d) {
          u[d] = velocity.value(x_q, d);
        }

        const auto& u_dot_n = scalar_product(u, n);
        const auto& lambda  = std::abs(u_dot_n);

        for(unsigned int i = 0; i < n_dofs; ++i) {
          copy_data_face.cell_rhs(i) -= fe_iv.jump_in_shape_values(i, q_index)*
                                        (dt*(u_dot_n*old_solution_average_values[q_index] + 0.5*lambda*old_solution_jump_values[q_index]))*
                                        fe_iv.JxW(q_index);
        }
      }
    };

    /*--- Compute single boundary face contribution ---*/
    const auto boundary_worker = [&](const Iterator&                    cell,
                                     const unsigned int&                f,
                                     MeshWorker::ScratchData<dim, dim>& scratch_data,
                                     CopyData&                          copy_data) {};

    /*--- Auxiliary lambda function to copy data from local to global ---*/
    const auto copier = [&](const CopyData& c) {
      constraints.distribute_local_to_global(c.cell_rhs,
                                             c.local_dof_indices,
                                             system_rhs_host);
      for(auto& cdf : c.face_data) {
        constraints.distribute_local_to_global(cdf.cell_rhs,
                                               cdf.joint_dof_indices,
                                               system_rhs_host);
      }
    };

    /*--- Create the needed auxiliary structues (i.e. quadrature formulas, scratch data and copy data) ---*/
    const QGauss<dim> quadrature_formula_cell(fe_degree + 1);
    const QGauss<dim - 1> quadrature_formula_face(fe_degree + 1);
    MeshWorker::ScratchData<dim, dim> scratch_data(fe,
                                                   quadrature_formula_cell,
                                                   update_values | update_gradients | update_quadrature_points | update_JxW_values,
                                                   quadrature_formula_face,
                                                   update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
    CopyData copy_data;

    /*--- Perform the loop that effectively builds the rhs ---*/
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);

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

    solution_host.update_ghost_values();
    data_out.add_data_vector(dof_handler, solution_host, "solution");
    data_out.build_patches(MappingQ1<dim>());

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    std::string output_file = "./" + saving_dir + "/solution_" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output_file, MPI_COMM_WORLD);
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

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_error_rho << error_L2_rho     << std::endl;
      output_error_rho << error_rel_L2_rho << std::endl;
    }
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
int main(int argc, char *argv[]) {
  try
  {
    using namespace AdvectionSolver;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    int         n_devices        = 0;
    cudaError_t cuda_error_code  = cudaGetDeviceCount(&n_devices);
    AssertCuda(cuda_error_code);
    const unsigned int my_mpi_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const int device_id          = my_mpi_id % n_devices;
    cuda_error_code              = cudaSetDevice(device_id);
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
