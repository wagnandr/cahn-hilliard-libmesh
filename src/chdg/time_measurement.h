#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace chdg {

class TimeMeasurement
{
 public:
   TimeMeasurement( const std::string& name );

   ~TimeMeasurement();

   TimeMeasurement( const TimeMeasurement& ) = delete;
   void operator=( const TimeMeasurement& ) = delete;
   TimeMeasurement( const TimeMeasurement&& ) = delete;
   void operator=( const TimeMeasurement&& ) = delete;

 private:
   std::string d_name;
   std::chrono::steady_clock::time_point d_begin;

   static std::size_t indent;
};

} // namespace chdg
