/*
 * Combining multiple frames together for multicamera setups
 * framebundle.h
 */

#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include "vilib/common/frame.h"

namespace vilib {

class FrameBundle {
public:
  FrameBundle(void) = default;
  FrameBundle(const std::vector<std::shared_ptr<Frame>> & frames);
  ~FrameBundle(void) = default;

  inline std::size_t id(void) const {
    return bundle_id_;
  }

  inline bool empty(void) const {
    return frames_.empty();
  }

  inline std::size_t size(void) const {
    return frames_.size();
  }

  inline const std::shared_ptr<Frame> & at(std::size_t i) const {
    return frames_[i];
  }

  inline std::shared_ptr<Frame> & at(std::size_t i) {
    return frames_[i];
  }
private:
  static std::size_t getNewId(void);

  std::size_t bundle_id_;
  std::vector<std::shared_ptr<Frame>> frames_;

  static std::size_t last_id_;
  static std::mutex last_id_mutex_;
};

} // namespace vilib
