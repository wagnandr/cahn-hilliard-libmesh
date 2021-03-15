#include "time_measurement.h"

namespace chdg {

TimeMeasurement::TimeMeasurement( const std::string& name )
: d_name( name )
, d_begin( std::chrono::steady_clock::now() )
{
   std::string ws;
   for ( std::size_t i = 0; i < indent; i += 1 )
      ws += ".";
   std::cout << ws << "start " << d_name << std::endl;

   indent += 1;
}

TimeMeasurement::~TimeMeasurement()
{
   indent -= 1;

   std::string ws;
   for ( std::size_t i = 0; i < indent; i += 1 )
      ws += ".";

   const auto end = std::chrono::steady_clock::now();
   const std::chrono::duration< float > elapsed = end - d_begin;

   std::cout << ws << "end " << d_name << " (took " << std::scientific << elapsed.count() << "s)" << std::endl;
}

std::size_t TimeMeasurement::indent = 0;

} // namespace chdg