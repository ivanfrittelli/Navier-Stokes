#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;


template <int dim, int spacedim>
struct NavierStokesParameters
{
  NavierStokesParameters()
    : initial_velocity_field(spacedim + 1),
    exact_solution(spacedim + 1),
    rhs_function(spacedim + 1),
    rhs_function_prev_time_step(spacedim+1),
    convergence_table(spacedim == 2 ?
                          std::vector<std::string>({{"u", "u", "p"}}) :
                          std::vector<std::string>({{"u", "u", "u", "p"}}),
                        {{VectorTools::H1_norm, VectorTools::L2_norm},
                         {VectorTools::L2_norm}})
  {
    prm.enter_subsection("Physical constants");
    {
      prm.add_parameter("Viscosity", eta); 
    }
    prm.leave_subsection();

    prm.enter_subsection("Functions");
    {
      prm.add_parameter("Initial velocity field", initial_velocity_field_expression);
      prm.add_parameter("Exact solution", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
      prm.add_parameter("RHS time dependent", rhs_time_dependent);
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Time step length", time_step_length);
      prm.add_parameter("Theta", theta);
    }
    prm.leave_subsection();

    prm.enter_subsection("Query parameters");
    {
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Number of time steps", number_of_steps);
      prm.add_parameter("Refine percentage", refine_percentage);
      prm.add_parameter("Coarsening percentage", coarsening_percentage);
      prm.add_parameter("Number of time cycles", number_of_time_cycles);
    }
    prm.leave_subsection();

    prm.enter_subsection("Output");
    {
      prm.add_parameter("File name", filename);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linearization parameters");
    {
      prm.add_parameter("Picard", picard);
      prm.add_parameter("Picard to Newton threshold", picard_to_newton_threshold);
      prm.add_parameter("Tollerance", tollerance);
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence table");
    convergence_table.add_parameters(prm);
    prm.leave_subsection();

    try
      {
        prm.parse_input("navier_stokes_" + std::to_string(spacedim) + "d.prm");
      }
    catch (std::exception &exc)
      {
        prm.print_parameters("navier_stokes_" + std::to_string(spacedim) + "d.prm");
        prm.parse_input("navier_stokes_" + std::to_string(spacedim) + "d.prm");
      }
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;
    constants["eta"] = eta;
    initial_velocity_field.initialize(FunctionParser<spacedim>::default_variable_names(),
                              {initial_velocity_field_expression},
                              constants);

    exact_solution.initialize(FunctionParser<spacedim>::default_variable_names()+",t", {exact_solution_expression}, constants, true);

    if (rhs_time_dependent) {
      rhs_function.initialize(FunctionParser<spacedim>::default_variable_names()+",t",
                            {rhs_expression},
                            constants, true);

      rhs_function_prev_time_step.initialize(FunctionParser<spacedim>::default_variable_names()+",t",
                            {rhs_expression},
                            constants, true);
    }

    else {
      rhs_function.initialize(FunctionParser<spacedim>::default_variable_names(),
                            {rhs_expression},
                            constants, false);

    }
    
  }
  double eta = 1.0;

  unsigned int fe_degree          = 1;
  unsigned int initial_refinement = 3;
  double theta = 0.5;
  mutable double time_step_length = 0.1;

  unsigned int n_cycles           = 1;
  mutable int number_of_steps = 5;
  double refine_percentage = 1;
  double coarsening_percentage = 0;
  int number_of_time_cycles = 1;

  bool picard = false;
  double picard_to_newton_threshold = 1e-3;
  double tollerance = 1e-12;

  std::string                  initial_velocity_field_expression = "if(y>0.9999999,1,0); 0; 0";
  std::string                  rhs_expression            = "0; 0; 0";
  std::string                  exact_solution_expression            = "if(y>0.9999999,1,0); 0; 0";
  bool rhs_time_dependent            = false;

  FunctionParser<spacedim> initial_velocity_field;
  mutable FunctionParser<spacedim> exact_solution;
  mutable FunctionParser<spacedim> rhs_function;
  mutable FunctionParser<spacedim> rhs_function_prev_time_step;

  std::string filename = "solution";

  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};



template <int dim, int spacedim>
class NavierStokes
{
public:
  NavierStokes(const NavierStokesParameters<dim, spacedim> &parameters);
  void
  run();

private:
  void
  make_grid();
  void
  estimate();
  void
  mark();
  void
  refine();
  void
  setup_system();
  void
  setup_constraints();
  void
  assemble_system_and_solve();
  void
  theta_time_step();
  void
  imex_time_step();
  void
  solve();
  void
  output_results(const unsigned int cycle) const;

  const NavierStokesParameters<dim, spacedim> &par;

  Triangulation<dim, spacedim> triangulation;
  FESystem<dim, spacedim>      fe;
  DoFHandler<dim, spacedim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  unsigned int cycle;

  Vector<float> estimated_error_per_cell;

  FEValuesExtractors::Vector velocity;
  FEValuesExtractors::Scalar pressure;

  Vector<double> old_time_solution;
};


template <int dim, int spacedim>
NavierStokes<dim, spacedim>::NavierStokes(const NavierStokesParameters<dim, spacedim> &par)
  : par(par)
  , fe(FE_Q<dim, spacedim>(par.fe_degree+1), spacedim, FE_Q<dim, spacedim>(par.fe_degree), 1)
  , dof_handler(triangulation)
  , velocity(0)
  , pressure(dim)
{
  
}



template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::make_grid()
{
  triangulation.clear();
 
  if (dim == spacedim)
    GridGenerator::hyper_cube(triangulation, 0, 1, false);
  else
    GridGenerator::torus<dim>(triangulation, 1.618, 1);

  triangulation.refine_global(par.initial_refinement);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}



template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::estimate()
{
  KellyErrorEstimator<dim, spacedim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 2),
                                     {},
                                     solution,
                                     estimated_error_per_cell);
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::mark()
{
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  par.refine_percentage,
                                                  par.coarsening_percentage);
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::refine()
{
  triangulation.execute_coarsening_and_refinement();
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  estimated_error_per_cell.reinit(triangulation.n_active_cells());

  old_time_solution.reinit(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, 
                          par.initial_velocity_field, 
                          old_time_solution);

  solution = old_time_solution;
  output_results(0);

  par.exact_solution.set_time(0);
  if(par.rhs_time_dependent) {
    //Porto indietro di uno
    par.rhs_function_prev_time_step.set_time(-par.time_step_length);
    par.rhs_function.set_time(0);
  }
}

template <int dim, int spacedim>
void 
NavierStokes<dim, spacedim>::setup_constraints() {
  constraints.clear();

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  
  if(dim == spacedim)
    VectorTools::interpolate_boundary_values(dof_handler,
                                          0,
                                          par.exact_solution,
                                          constraints, fe.component_mask(velocity));

  constraints.add_line(spacedim);

  constraints.close();
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::assemble_system_and_solve()
{
  for (int time_step = 1; time_step < par.number_of_steps + 1; time_step++) {
    par.exact_solution.advance_time(par.time_step_length);

    if(par.rhs_time_dependent) {
      par.rhs_function.advance_time(par.time_step_length);
      par.rhs_function_prev_time_step.advance_time(par.time_step_length);
    }
    
    setup_constraints();

    std::cout<<"Time step " << time_step << "\n";

    theta_time_step();

    old_time_solution = solution;
    output_results(time_step);
  }

}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::theta_time_step() {
  QGauss<dim>     quadrature_formula(fe.degree + 2);

  FEValues<dim, spacedim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //Variabili per time step precedente
  std::vector<Tensor<1,spacedim>> old_time_function_values(fe_values.n_quadrature_points);
  std::vector<Tensor<2,spacedim>> old_time_function_gradients(fe_values.n_quadrature_points);

  std::vector<double> old_time_pressure_values(fe_values.n_quadrature_points);
  
  //Variabili per Newton step precedente
  double residue = 0;
  Vector<double> old_newton_solution(dof_handler.n_dofs());
  bool using_picard = par.picard;

  std::vector<Tensor<1,spacedim>> old_newton_function_values(fe_values.n_quadrature_points);
  std::vector<Tensor<2,spacedim>> old_newton_function_gradients(fe_values.n_quadrature_points);
  std::vector<double> old_newton_function_divergences(fe_values.n_quadrature_points);

  std::vector<double> old_newton_pressure_values(fe_values.n_quadrature_points);

  Vector<double> residue_vector(dof_handler.n_dofs());
  Vector<double> cell_residue_vector(dofs_per_cell);

  //Variabili che memorizzano il valore di u, grad u, div u, p in modo da calcolarlo n volte invece che n^2
  std::vector<Tensor<1,spacedim>> u(dofs_per_cell);
  std::vector<Tensor<2,spacedim>> grad_u(dofs_per_cell);
  std::vector<double> div_u(dofs_per_cell);
  std::vector<double> p(dofs_per_cell);

  old_newton_solution = old_time_solution;

  do {
    system_matrix = 0;
    system_rhs=0;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
              
      fe_values[velocity].get_function_values(old_time_solution, old_time_function_values);
      fe_values[velocity].get_function_gradients(old_time_solution, old_time_function_gradients);
      fe_values[pressure].get_function_values(old_time_solution, old_time_pressure_values);

      fe_values[velocity].get_function_values(old_newton_solution, old_newton_function_values);
      fe_values[velocity].get_function_gradients(old_newton_solution, old_newton_function_gradients);

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        double JxW = fe_values.JxW(q_index);

        for (unsigned int i = 0; i < dofs_per_cell; i++) {
          u[i] = fe_values[velocity].value(i, q_index);
          grad_u[i] = fe_values[velocity].gradient(i, q_index);
          div_u[i] = fe_values[velocity].divergence(i, q_index);
          p[i] = fe_values[pressure].value(i, q_index);
        }
                      
        const auto &x_q = fe_values.quadrature_point(q_index);

        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            const unsigned int comp_i = fe.system_to_component_index(i).first;

            const auto v_i      = u[i]; //fe_values[velocity].value(i, q_index);
            const auto div_v_i  = div_u[i];//fe_values[velocity].divergence(i, q_index);
            const auto grad_v_i = grad_u[i];//fe_values[velocity].gradient(i, q_index)
            const auto q_i = p[i];//fe_values[pressure].value(i, q_index);

            //I termini simmetrici li metto sulla matrice solo una volta (massa-diffusione)
            for(unsigned int j = 0; j <= i; j++) {
              double to_sum_cell_matrix = 0;

              const auto u_j      = u[j]; //fe_values[velocity].value(j, q_index);
              const auto grad_u_j = grad_u[j];//fe_values[velocity].gradient(j, q_index);
              const auto div_u_j = div_u[j];//fe_values[velocity].divergence(j, q_index);
              const auto p_j     = p[j];//fe_values[pressure].value(j, q_index);

              to_sum_cell_matrix +=
                  scalar_product(u_j, v_i);

              to_sum_cell_matrix += par.theta * par.time_step_length *
                  (par.eta * scalar_product(grad_u_j, grad_v_i)); 

              to_sum_cell_matrix -= par.time_step_length * (div_u_j * q_i);
              to_sum_cell_matrix -= par.time_step_length * (div_v_i * p_j);

              cell_matrix(i,j) += to_sum_cell_matrix*JxW;

              //La diagonale non va sommata due volte
              if(i != j)
                cell_matrix(j, i) += to_sum_cell_matrix * JxW;
            }

            //Quelli non simmetrici due volte (trasporto)
          for (unsigned int j = 0; j < dofs_per_cell; j++)
            {
              double to_sum_cell_matrix = 0;

              const auto u_j      = u[j]; //fe_values[velocity].value(j, q_index);
              const auto grad_u_j_1 = grad_u[j];//fe_values[velocity].gradient(j, q_index);              

              //termine di trasporto linearizzato
              to_sum_cell_matrix += par.theta *  par.time_step_length * grad_u_j_1 * old_newton_function_values[q_index]*v_i;

              //Non appare in Picard
              if(!using_picard)
                to_sum_cell_matrix += par.theta * par.time_step_length * old_newton_function_gradients[q_index]*u_j*v_i;

              cell_matrix(i,j) += to_sum_cell_matrix*JxW;

            }

            double to_sum_rhs = 0;

            to_sum_rhs += scalar_product(old_time_function_values[q_index], v_i);

            to_sum_rhs -= (1-par.theta) * par.time_step_length 
                  * (par.eta * scalar_product(old_time_function_gradients[q_index], grad_v_i));

            to_sum_rhs -= (1-par.theta) * par.time_step_length *
                            old_time_function_gradients[q_index] *
                            old_time_function_values[q_index]*v_i;     

            //Non appare in Picard
            if(!using_picard)
              to_sum_rhs += par.theta* par.time_step_length * old_newton_function_gradients[q_index]*old_newton_function_values[q_index]*v_i;

            if (comp_i < spacedim) {
              if (par.rhs_time_dependent)
                to_sum_rhs += par.time_step_length * (v_i[comp_i] * // phi_i(x_q)
                  ((1 - par.theta) * par.rhs_function_prev_time_step.value(x_q, comp_i) + par.theta * par.rhs_function.value(x_q, comp_i))
                );    

              else 
                to_sum_rhs += par.time_step_length * (v_i[comp_i] * // phi_i(x_q)
                    par.rhs_function.value(x_q, comp_i)
                  );  
            }
    

            cell_rhs(i) += to_sum_rhs * JxW;
          }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

    solve();

    old_newton_solution = solution;

    residue_vector=0;

    //Calcolo residuo
    for(const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);

      cell_residue_vector = 0;

      //un
      fe_values[velocity].get_function_values(old_time_solution, old_time_function_values);
      fe_values[velocity].get_function_gradients(old_time_solution, old_time_function_gradients);

      //un+1
      fe_values[velocity].get_function_values(old_newton_solution, old_newton_function_values);
      fe_values[velocity].get_function_gradients(old_newton_solution, old_newton_function_gradients);
      fe_values[velocity].get_function_divergences(old_newton_solution, old_newton_function_divergences);

      fe_values[pressure].get_function_values(old_newton_solution, old_newton_pressure_values);

      
      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        const auto &x_q = fe_values.quadrature_point(q_index);
        
        double JxW = fe_values.JxW(q_index);

        for (const unsigned int i : fe_values.dof_indices()) 
        {
          double to_sum_cell_residue = 0;

          const unsigned int comp_i = fe.system_to_component_index(i).first;

          const auto v_i      = fe_values[velocity].value(i, q_index);
          const auto div_v_i  = fe_values[velocity].divergence(i, q_index);
          const auto grad_v_i = fe_values[velocity].gradient(i, q_index);

          const auto q_i = fe_values[pressure].value(i, q_index);

          to_sum_cell_residue +=
            scalar_product(old_newton_function_values[q_index], v_i);

          //(costanti * (nabla u_n+1, nabla v))
          to_sum_cell_residue += par.theta * par.time_step_length *
            (par.eta * scalar_product(old_newton_function_gradients[q_index], grad_v_i));                    

          //termine di trasporto
          to_sum_cell_residue += par.theta * par.time_step_length * old_newton_function_gradients[q_index] * old_newton_function_values[q_index]*v_i;
          to_sum_cell_residue += (1-par.theta) * par.time_step_length * old_time_function_gradients[q_index] * old_time_function_values[q_index]*v_i;

          //(div u_n+1, q) - (div v, p_n+1)
          to_sum_cell_residue -= par.time_step_length * (old_newton_function_divergences[q_index] * q_i);
          to_sum_cell_residue -= par.time_step_length * (div_v_i * old_newton_pressure_values[q_index]);

          to_sum_cell_residue -= scalar_product(old_time_function_values[q_index], v_i) ;
          to_sum_cell_residue += (1-par.theta) * par.time_step_length 
                * (par.eta * scalar_product(old_time_function_gradients[q_index], grad_v_i));

          if (comp_i < spacedim) {
            if(par.rhs_time_dependent)
              to_sum_cell_residue -= par.time_step_length * (v_i[comp_i] * // phi_i(x_q)
                ((1 - par.theta) * par.rhs_function_prev_time_step.value(x_q, comp_i) + par.theta * par.rhs_function.value(x_q, comp_i))); // f(x_q));              // dx
            
            else
              to_sum_cell_residue -= par.time_step_length * (v_i[comp_i] * // phi_i(x_q)
                par.rhs_function.value(x_q, comp_i)); // f(x_q));              // dx
            
          }
              
          cell_residue_vector(i) += to_sum_cell_residue * JxW;
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_residue_vector, local_dof_indices, residue_vector);
    }

    residue = residue_vector.l2_norm();
    std::cout << "Residue: " << residue << "\n";
    
    if (using_picard && residue < par.picard_to_newton_threshold) using_picard = false;

  } while(residue > par.tollerance);

  solution = old_newton_solution;
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::imex_time_step() {
  QGauss<dim>     quadrature_formula(fe.degree + 2);

  FEValues<dim, spacedim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //Variabili per time step precedente
  std::vector<Tensor<1,spacedim>> old_time_function_values(fe_values.n_quadrature_points);
  std::vector<Tensor<2,spacedim>> old_time_function_gradients(fe_values.n_quadrature_points);

  std::vector<double> old_time_pressure_values(fe_values.n_quadrature_points);

  //Variabili che memorizzano il valore di u, grad u, div u, p in modo da calcolarlo n volte invece che n^2
  std::vector<Tensor<1,spacedim>> u(dofs_per_cell);
  std::vector<Tensor<2,spacedim>> grad_u(dofs_per_cell);
  std::vector<double> div_u(dofs_per_cell);
  std::vector<double> p(dofs_per_cell);

  system_matrix = 0;
  system_rhs=0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;
            
    fe_values[velocity].get_function_values(old_time_solution, old_time_function_values);
    fe_values[velocity].get_function_gradients(old_time_solution, old_time_function_gradients);
    fe_values[pressure].get_function_values(old_time_solution, old_time_pressure_values);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      double JxW = fe_values.JxW(q_index);

      for (unsigned int i = 0; i < dofs_per_cell; i++) {
        u[i] = fe_values[velocity].value(i, q_index);
        grad_u[i] = fe_values[velocity].gradient(i, q_index);
        div_u[i] = fe_values[velocity].divergence(i, q_index);
        p[i] = fe_values[pressure].value(i, q_index);
      }
                    
      const auto &x_q = fe_values.quadrature_point(q_index);

      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          const unsigned int comp_i = fe.system_to_component_index(i).first;

          const auto v_i      = u[i]; //fe_values[velocity].value(i, q_index);
          const auto div_v_i  = div_u[i];//fe_values[velocity].divergence(i, q_index);
          const auto grad_v_i = grad_u[i];//fe_values[velocity].gradient(i, q_index)
          const auto q_i = p[i];//fe_values[pressure].value(i, q_index);

          //I termini simmetrici li metto sulla matrice solo una volta (massa-diffusione)
          for(unsigned int j = 0; j <= i; j++) {
            double to_sum_cell_matrix = 0;

            const auto u_j      = u[j]; //fe_values[velocity].value(j, q_index);
            const auto grad_u_j = grad_u[j];//fe_values[velocity].gradient(j, q_index);
            const auto div_u_j = div_u[j];//fe_values[velocity].divergence(j, q_index);
            const auto p_j     = p[j];//fe_values[pressure].value(j, q_index);

            to_sum_cell_matrix +=
                scalar_product(u_j, v_i);

            to_sum_cell_matrix += par.theta * par.time_step_length *
                (par.eta * scalar_product(grad_u_j, grad_v_i)); 

            to_sum_cell_matrix -= par.time_step_length * (div_u_j * q_i);
            to_sum_cell_matrix -= par.time_step_length * (div_v_i * p_j);

            cell_matrix(i,j) += to_sum_cell_matrix*JxW;

            //La diagonale non va sommata due volte
            if(i != j)
              cell_matrix(j, i) += to_sum_cell_matrix * JxW;
          }

          //Quelli non simmetrici due volte (trasporto)
        for (unsigned int j = 0; j < dofs_per_cell; j++)
          {
            double to_sum_cell_matrix = 0;

            const auto grad_u_j_1 = grad_u[j];//fe_values[velocity].gradient(j, q_index);              

            //termine di trasporto linearizzato
            to_sum_cell_matrix += par.theta *  par.time_step_length * grad_u_j_1 * old_time_function_values[q_index]*v_i;

            cell_matrix(i,j) += to_sum_cell_matrix*JxW;

          }

          double to_sum_rhs = 0;

          to_sum_rhs += scalar_product(old_time_function_values[q_index], v_i);

          if (comp_i < spacedim)
            to_sum_rhs += par.time_step_length * (v_i[comp_i] * // phi_i(x_q)
              par.rhs_function.value(x_q, comp_i)
            );  
          
          cell_rhs(i) += to_sum_rhs * JxW;
        }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
      cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  solve();
}

template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::solve()
{
  SparseDirectUMFPACK solver;
  solver.initialize(system_matrix);
  solver.vmult(solution, system_rhs);

  constraints.distribute(solution);
}



template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::output_results(const unsigned int time_step) const
{
  DataOut<dim, spacedim> data_out;

  std::vector<std::string> names(spacedim + 1, "u");
  names[spacedim] = "p"; // last component is pressure

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      spacedim, DataComponentInterpretation::component_is_part_of_vector);
  component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           names,
                           DataOut<dim, spacedim>::type_dof_data,
                           component_interpretation);

  data_out.build_patches();

  auto fname =
    par.filename + "-" + std::to_string(dim) + "d_" + std::to_string(cycle) + "_"  + std::to_string(time_step) + ".vtu";

  std::ofstream output(fname);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back({cycle, fname});

  std::ofstream pvd_output(par.filename + "-" + std::to_string(dim) + "d_" + std::to_string(cycle) + ".pvd");

  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}



template <int dim, int spacedim>
void
NavierStokes<dim, spacedim>::run()
{
  std::cout << "Solving problem in " << dim << " dimension and " << spacedim << " space dimension."
            << std::endl;

  for(int i = 0; i < par.number_of_time_cycles; i++) {
    for (cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        {
          mark();
          refine();
        }
      setup_system();
      assemble_system_and_solve();
      estimate();
      par.convergence_table.error_from_exact(dof_handler,
                                             solution,
                                             par.exact_solution);
    }
    
    par.time_step_length/=2;
    //par.number_of_steps*=2;
  }
  
  par.convergence_table.output_table(std::cout);
}



int
main()
{
  {
    NavierStokesParameters<2, 3> par;
    NavierStokes<2, 3>           navier_stokes_2d(par);
    navier_stokes_2d.run();
  }

  return 0;
}