/*
 * Statistics class storing and organizing statistical data
 * statistics.h
 */

#pragma once

#include <string>

namespace vilib {

class Statistics {
public:
  Statistics(const char * unit, const char * name = NULL, int indent = 1);
  ~Statistics(void)=default;

  void add(double data);
  void reset(void);
  inline const double & get_min(void) const { return min_; }
  inline const double & get_max(void) const { return max_; }
  inline double get_avg(void) const { return avg_/avg_count_; }
  inline const std::string & get_name(void) const { return name_; }
  inline const std::string & get_unit(void) const { return unit_; }
  void display(int decimal_places=0,double scale=1.0,const char * new_unit=NULL) const;

private:
  double min_;
  double max_;
  double avg_;
  std::size_t avg_count_;
  std::string name_;
  std::string unit_;
  int indent_;
};

} // namespace vilib
