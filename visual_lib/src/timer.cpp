/*
 * Timer class for profiling algorithms
 * timer.cpp
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

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include "vilib/timer.h"

namespace vilib {

Timer::Timer(const char * name, int indent) :
  name_((name==nullptr)?"Unnamed timer":name),time_(0.0),indent_(indent),stat_("usec") {
}

Timer::Timer(const std::string & name, int indent) :
  name_(name),time_(0.0),indent_(indent),stat_("usec") {
}

void Timer::start(void) {
  time_ = -1.0*this->get_localtime_usec();
}

void Timer::stop(void) {
  time_ += this->get_localtime_usec();
}

void Timer::pause(void) {
  time_ += this->get_localtime_usec();
}

void Timer::cont(void) {
  time_ -= this->get_localtime_usec();
}

double Timer::get_localtime_usec(void) const {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec*1.0e6 + (double)t.tv_usec;
}

void Timer::reset(void) {
  time_ = 0.0;
}

void Timer::add_to_stat_n_reset(void) {
  stat_.add(time_);
  time_ = 0.0;
}

void Timer::display_usec(void) const {
  for(int i=0;i<indent_;++i) {
    std::cout << " ";
  }
  std::cout << std::setw(name_size_characters_) << std::left << name_ << ": "
            << std::setw(time_size_characters_) << std::right << time_ << " usec" << std::endl;
}

void Timer::display_stat_usec(void) const {
  for(int i=0;i<indent_;++i) {
    std::cout << " ";
  }
  std::cout << std::setw(name_size_characters_) << std::left  << name_ << ": ";
  stat_.display();
}

} // namespace vilib
