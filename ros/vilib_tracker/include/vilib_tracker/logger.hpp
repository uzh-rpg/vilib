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

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

namespace vilib {

namespace {
struct NoPrint {
  template<typename T>
  constexpr NoPrint operator<<(const T &) const noexcept {
    return NoPrint();
  }
  constexpr NoPrint operator<<(std::ostream &(*)(std::ostream &)) const  //
    noexcept {
    return NoPrint();
  }
};
}  // namespace

template<typename OutputStream>
class LoggerBase {
 public:
  LoggerBase(const std::string &name, const bool color = true)
    : name_(padName(name)), colored_(color) {
    sink_ = std::make_unique<std::ostream>(std::cout.rdbuf());
    sink_->precision(DEFAULT_PRECISION);
  }

  inline std::streamsize precision(const std::streamsize n) {
    return sink_->precision(n);
  }
  inline void scientific(const bool on = true) {
    *sink_ << (on ? std::scientific : std::fixed);
  }

  static constexpr int MAX_CHARS = 256;

  void info(const char *msg, ...) const {
    std::va_list args;
    va_start(args, msg);
    char buf[MAX_CHARS];
    const int n = std::vsnprintf(buf, MAX_CHARS, msg, args);
    va_end(args);
    if (n < 0 || n >= MAX_CHARS)
      *sink_ << name_ << "=== Logging error ===" << std::endl;
    if (colored_)
      *sink_ << name_ << buf << std::endl;
    else
      *sink_ << name_ << INFO << buf << std::endl;
  }

  void warn(const char *msg, ...) const {
    std::va_list args;
    va_start(args, msg);
    char buf[MAX_CHARS];
    const int n = std::vsnprintf(buf, MAX_CHARS, msg, args);
    va_end(args);
    if (n < 0 || n >= MAX_CHARS)
      *sink_ << name_ << "=== Logging error ===" << std::endl;
    if (colored_)
      *sink_ << YELLOW << name_ << buf << RESET << std::endl;
    else
      *sink_ << name_ << WARN << buf << std::endl;
  }

  void error(const char *msg, ...) const {
    std::va_list args;
    va_start(args, msg);
    char buf[MAX_CHARS];
    const int n = std::vsnprintf(buf, MAX_CHARS, msg, args);
    va_end(args);
    if (n < 0 || n >= MAX_CHARS)
      *sink_ << name_ << "=== Logging error ===" << std::endl;
    if (colored_)
      *sink_ << RED << name_ << buf << RESET << std::endl;
    else
      *sink_ << name_ << ERROR << buf << std::endl;
  }

  void fatal(const char *msg, ...) const {
    std::va_list args;
    va_start(args, msg);
    char buf[MAX_CHARS];
    const int n = std::vsnprintf(buf, MAX_CHARS, msg, args);
    va_end(args);
    if (n < 0 || n >= MAX_CHARS)
      *sink_ << name_ << "=== Logging error ===" << std::endl;
    if (colored_)
      *sink_ << RED << name_ << buf << RESET << std::endl;
    else
      *sink_ << name_ << FATAL << buf << std::endl;
    throw std::runtime_error(name_ + buf);
  }

#ifdef DEBUG_LOG
  void debug(const char *msg, ...) const {
    std::va_list args;
    va_start(args, msg);
    char buf[MAX_CHARS];
    const int n = std::vsnprintf(buf, MAX_CHARS, msg, args);
    va_end(args);
    if (n < 0 || n >= MAX_CHARS)
      *sink_ << name_ << "=== Logging error ===" << std::endl;
    *sink_ << name_ << buf << std::endl;
  }
  OutputStream &debug() const { return *sink_ << name_; }
  constexpr void debug(const std::function<void(void)> &&lambda) const {
    lambda();
  }

  static constexpr bool debugEnabled() { return true; }
#else
  inline constexpr void debug(const char *, ...) const noexcept {}
  inline constexpr NoPrint debug() const { return NoPrint(); }
  inline constexpr void debug(const std::function<void(void)> &&) const  //
    noexcept {}
  static constexpr bool debugEnabled() { return false; }
#endif

  template<typename T>
  OutputStream &operator<<(const T &printable) const {
    return *sink_ << name_ << printable;
  }

  OutputStream &operator<<(OutputStream &(*printable)(OutputStream &)) const {
    return *sink_ << name_ << printable;
  }

  inline void newline(const int n = 1) {
    *sink_ << std::string((size_t)n, '\n');
  }

  inline const std::string &name() const { return name_; }

 protected:
  LoggerBase(const std::string &name, const bool color,
             std::shared_ptr<OutputStream> sink)
    : name_(padName(name)), colored_(color), sink_(sink) {}

  static std::string padName(const std::string &name) {
    if (name.empty()) return "";
    const std::string padded = "[" + name + "] ";
    const int extra = LoggerBase::NAME_PADDING - (int)padded.size();
    return extra > 0 ? padded + std::string((size_t)extra, ' ') : padded;
  }

  static constexpr int DEFAULT_PRECISION = 3;
  static constexpr int NAME_PADDING = 20;
  static constexpr char RESET[] = "\033[0m";
  static constexpr char RED[] = "\033[31m";
  static constexpr char YELLOW[] = "\033[33m";
  static constexpr char INFO[] = "Info:    ";
  static constexpr char WARN[] = "Warning: ";
  static constexpr char ERROR[] = "Error:   ";
  static constexpr char FATAL[] = "Fatal:   ";

  const std::string name_;
  const bool colored_;
  std::shared_ptr<OutputStream> sink_;
};

using Logger = LoggerBase<std::ostream>;

}  // namespace vilib
