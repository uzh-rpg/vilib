/*
 * Statistics class storing and organizing statistical data
 * statistics.h
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
