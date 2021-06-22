/*
 * Copyright (c) 2019-2021 Philipp Foehn,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace vilib {

template<typename T = double>
class Statistic {
 public:
  Statistic(const std::string &name = "Statistic",
            const int max_samples = std::numeric_limits<int>::max())
    : name_(name), max_samples_(max_samples) {}
  Statistic(const Statistic &rhs) = default;
  Statistic &operator=(const Statistic &rhs) {
    n_ = rhs.n_;
    mean_ = rhs.mean_;
    last_ = rhs.last_;
    S_ = rhs.S_;
    min_ = rhs.min_;
    max_ = rhs.max_;
    return *this;
  }

  T operator<<(const T in) {
    if (!std::isfinite(in)) return in;

    mean_ = n_ ? mean_ : in;
    last_ = in;
    const T mean_last = mean_;
    n_ += n_ < max_samples_ ? 1 : 0;
    mean_ += (in - mean_last) / (T)(n_);
    S_ += (in - mean_last) * (in - mean_);
    min_ = std::min(in, min_);
    max_ = std::max(in, max_);

    return mean_;
  }

  T add(const T in) { return operator<<(in); }

  T operator()() const { return mean_; }
  operator double() const { return (double)mean_; }
  operator float() const { return (float)mean_; }
  operator int() const { return n_; }

  int count() const { return n_; }
  T last() const { return last_; }
  T mean() const { return mean_; }
  T std() const { return n_ > 1 ? std::sqrt(S_ / (T)(n_ - 1)) : 0.0; }
  T min() const { return min_; }
  T max() const { return max_; }

  const std::string &name() const { return name_; }

  void reset() {
    n_ = 0;
    last_ = 0.0;
    mean_ = 0.0;
    S_ = 0.0;
    min_ = std::numeric_limits<T>::max();
    max_ = std::numeric_limits<T>::min();
  }

  friend std::ostream &operator<<(std::ostream &os, const Statistic &s) {
    if (s.n_ < 1) os << s.name_ << "has no sample yet!" << std::endl;

    const std::streamsize prec = os.precision();
    os.precision(3);

    os << std::left << std::setw(16) << s.name_ << "mean|std  ";
    os << std::left << std::setw(5) << s.mean() << "|";
    os << std::left << std::setw(5) << s.std() << "  [min|max:  ";
    os << std::left << std::setw(5) << s.min() << "|";
    os << std::left << std::setw(5) << s.max() << "]" << std::endl;

    os.precision(prec);
    return os;
  }

 protected:
  const std::string name_;
  const int max_samples_;
  int n_{0};
  T mean_{0.0};
  T last_{0.0};
  T S_{0.0};
  T min_{std::numeric_limits<T>::max()};
  T max_{std::numeric_limits<T>::min()};
};

}  // namespace vilib
