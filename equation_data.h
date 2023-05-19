/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial conditions,
// the velocity field
//
namespace EquationData {
  using namespace dealii;

  static const unsigned int degree = 2;   /*--- Polynomial degree ---*/

  // We declare now the class that describes the velocity.
  //
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;
  };


  // Constructor which simply relies on the 'Function' constructor.
  //
  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    const double omega = 1.0;
    const double x0    = 0.0;
    const double y0    = 0.0;

    if(component == 0) {
      return -omega*(p[1] - y0);
    }
    return omega*(p[0] - x0);
  }

  // Put together for a vector evalutation of the velocity.
  //
  template<int dim>
  void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function.
  //
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  template<int dim>
  Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}

  template<int dim>
  double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double x0    = 1.0/6.0;
    const double y0    = 1.0/6.0;
    const double sigma = 0.2;

    const double X     = (p[0] - x0)/sigma;
    const double Y     = (p[1] - y0)/sigma;

    return (0.25*(1.0 + std::cos(numbers::PI*X))*(1.0 + std::cos(numbers::PI*Y)))*((X*X + Y*Y) <= 1);
  }

} // namespace EquationData
