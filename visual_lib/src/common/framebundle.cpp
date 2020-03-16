/*
 * Combining multiple frames together for multicamera setups
 * framebundle.cpp
 */

#include "vilib/common/framebundle.h"

namespace vilib {

std::size_t FrameBundle::last_id_ = 0;
std::mutex FrameBundle::last_id_mutex_;

FrameBundle::FrameBundle(const std::vector<std::shared_ptr<Frame>> & frames) :
  bundle_id_(getNewId()),
  frames_(frames) {
}

std::size_t FrameBundle::getNewId(void) {
    std::lock_guard<std::mutex> lock(last_id_mutex_);
    return last_id_++;
}

} // namespace vilib