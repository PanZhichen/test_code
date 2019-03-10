#ifndef COLORS_H
#define COLORS_H
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>

namespace pcl
{

  PCL_EXPORTS RGB
  getRandomColor (double min = 0.2, double max = 2.8);

  enum ColorLUTName
  {
    /** Color lookup table consisting of 256 colors structured in a maximally
      * discontinuous manner. Generated using the method of Glasbey et al.
      * (see https://github.com/taketwo/glasbey) */
    LUT_GLASBEY,
    /** A perceptually uniform colormap created by Stéfan van der Walt and
      * Nathaniel Smith for the Python matplotlib library.
      * (see https://youtu.be/xAoljeRJ3lU for background and overview) */
    LUT_VIRIDIS,
  };

  template <ColorLUTName T>
  class ColorLUT
  {

    public:

      /** Get a color from the lookup table with a given id.
        *
        * The id should be less than the size of the LUT (see size()). */
      static RGB at (size_t color_id);

      /** Get the number of colors in the lookup table.
        *
        * Note: the number of colors is different from the number of elements
        * in the lookup table (each color is defined by three bytes). */
      static size_t size ();

      /** Get a raw pointer to the lookup table. */
      static const unsigned char* data ();

  };

  typedef ColorLUT<pcl::LUT_GLASBEY> GlasbeyLUT;
  typedef ColorLUT<pcl::LUT_VIRIDIS> ViridisLUT;

}
#endif /* PCL_COMMON_COLORS_H */
