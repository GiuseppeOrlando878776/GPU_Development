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

#include <deal.II/base/cuda.h>

#include <deal.II/lac/cuda_sparse_matrix.h>

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

    void run();

  private:
    void create_triangulation(const unsigned int n_refines);

    void setup_system();

    void initialize();

    void assemble_system();

    void solve();

    void output_results(const unsigned int step) const;

    Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    // Since all the operations in the `solve()` function are executed on the
    // graphics card, it is necessary for the vectors used to store their values
    // on the GPU as well. In addition, we also keep a solution vector with
    // CPU storage such that we can view and display the solution as usual.
    CUDAWrappers::SparseMatrix<double> system_matrix_dev;
    SparsityPattern sparsity_pattern;

    LinearAlgebra::CUDAWrappers::Vector<double> solution_dev;
    LinearAlgebra::CUDAWrappers::Vector<double> system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host,
                                                                  solution_host_old;

    ConditionalOStream pcout;

    EquationData::Density<dim>  density;
    EquationData::Velocity<dim> velocity;

    const double t_0; /*--- Initial time auxiliary variable ----*/
    const double T;   /*--- Final time auxiliary variable ----*/
    double       dt;  /*--- Time step auxiliary variable ---*/
    unsigned int n_refines;  /*--- Number of refinements auxiliary variable ---*/

    unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
    double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

    std::string  saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/
    unsigned int output_interval; /* --- Auxiliary variable for the number of iterations to save the results ---*/
  };


  // Class constructor
  //
  template <int dim, int fe_degree>
  AdvectionProblem<dim, fe_degree>::AdvectionProblem(RunTimeParameters::Data_Storage& data) : triangulation(),
                                                                                              fe(fe_degree),
                                                                                              dof_handler(triangulation),
                                                                                              pcout(std::cout),
                                                                                              density(),
                                                                                              velocity(),
                                                                                              t_0(data.initial_time),
                                                                                              T(data.final_time),
                                                                                              dt(data.dt),
                                                                                              n_refines(data.n_global_refines),
                                                                                              max_its(data.max_iterations),
                                                                                              eps(data.eps),
                                                                                              saving_dir(data.dir),
                                                                                              output_interval(data.output_interval) {}

  // Build the domain
  //
  template<int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::create_triangulation(const unsigned int n_refines) {
    GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, true);

    triangulation.refine_global(n_refines);
  }


  // Setup of the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::setup_system() {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
    DoFTools::make_periodicity_constraints(dof_handler, 2, 3, 1, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    solution_dev.reinit(dof_handler.n_dofs());
    system_rhs_dev.reinit(dof_handler.n_dofs());

    solution_host.reinit(dof_handler.n_dofs());
    solution_host_old.reinit(solution_host);
  }


  // Initialize the field
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::initialize() {
    VectorTools::interpolate(dof_handler, density, solution_host);

    LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.n_dofs());
    rw_vector.import(solution_host, VectorOperation::insert);
    solution_dev.import(rw_vector, VectorOperation::insert);
  }


  // We assemble now the matrix
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::assemble_system() {
    SparseMatrix<double> system_matrix_host;
    system_matrix_host.reinit(sparsity_pattern);

    Vector<double> system_rhs_host(dof_handler.n_dofs());

    const QGauss<dim> quadrature_formula(2*fe_degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> old_solution_values(n_q_points);
    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      cell_matrix = 0;
      cell_rhs    = 0;

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
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            cell_matrix(i,j) += fe_values.shape_value(i, q_index)*
                                (fe_values.shape_value(j, q_index))*
                                fe_values.JxW(q_index);
          }

          cell_rhs(i) += fe_values.shape_value(i, q_index)*
                         (old_solution_values[q_index] - dt*scalar_product(u, old_solution_gradients[q_index]))*
                         fe_values.JxW(q_index);
        }
      }

      cell->get_dof_indices(local_dof_indices);

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


  // Solve the system
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::solve() {
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
  void AdvectionProblem<dim, fe_degree>::output_results(const unsigned int step) const {
    DataOut<dim> data_out;

    solution_host.update_ghost_values();

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_host, "solution");
    data_out.build_patches();

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    std::ofstream output_file("./" + saving_dir + "/solution_" + Utilities::int_to_string(step, 5) + ".vtu");
    data_out.write_vtu(output_file);
  }


  // There is nothing surprising in the `run()` function either. We simply
  // compute the solution on a series of (globally) refined meshes.
  //
  template <int dim, int fe_degree>
  void AdvectionProblem<dim, fe_degree>::run() {
    create_triangulation(n_refines);
    setup_system();
    initialize();

    output_results(0);
    double time    = t_0;
    unsigned int n = 0;

    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      solution_host_old.equ(1.0, solution_host);

      assemble_system();
      solve();

      assemble_system();
      solve();
      solution_host *= 0.25;
      solution_host.add(0.75, solution_host_old);

      assemble_system();
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

    deallog.depth_console(2);

    int         n_devices       = 0;
    cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
    AssertCuda(cuda_error_code);
    cuda_error_code = cudaSetDevice(0);
    AssertCuda(cuda_error_code);

    AdvectionProblem<2, EquationData::degree> advection_problem(data);
    advection_problem.run();
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
