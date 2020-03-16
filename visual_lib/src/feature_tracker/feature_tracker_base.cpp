/*
 * Feature tracker base class
 * feature_tracker_base.cpp
 */

#include "vilib/feature_tracker/feature_tracker_base.h"

namespace vilib {

FeatureTrackerBase::FeatureTrackerBase(const FeatureTrackerOptions & options,
                                       const std::size_t & camera_num) :
  options_(options)
#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
  ,life_stat_("[1]","Feature track life")
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
{
  tracks_.resize(camera_num);
  tracked_features_num_.resize(camera_num,0);
  detected_features_num_.resize(camera_num,0);
}

FeatureTrackerBase::~FeatureTrackerBase(void) {
}

void FeatureTrackerBase::setDetectorCPU(std::shared_ptr<DetectorBase> & detector,
                                        const std::size_t & camera_id) {
  (void)detector;
  (void)camera_id;
  assert(0 && "This handler should have been overridden");
}

void FeatureTrackerBase::setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                                        const std::size_t & camera_id) {
  (void)detector;
  (void)camera_id;
  assert(0 && "This handler should have been overridden");
}

void FeatureTrackerBase::setBestNFeatures(int n) {
  options_.use_best_n_features = n;
}

void FeatureTrackerBase::setMinTracksToDetect(int n) {
  options_.min_tracks_to_detect_new_features = n;
}

void FeatureTrackerBase::removeTracks(const std::vector<std::size_t> & ind_to_remove,
                                      const std::size_t & camera_id) {
  /*
   * Note to future self:
   * the original idea came from SVO: container_helpers.h
   * and it was extended with std::move()
   */
  std::vector<struct FeatureTrack> & source = tracks_[camera_id];
  std::vector<struct FeatureTrack> dest;
  std::size_t start_offset;
  std::size_t end_offset;
  dest.reserve(source.size() - ind_to_remove.size());
  for(std::size_t i=0;i<ind_to_remove.size();++i) {
    start_offset = (i==0)?0:(ind_to_remove[i-1]+1);
    end_offset   = ind_to_remove[i];
    std::move(source.begin()+start_offset,source.begin()+end_offset,std::back_inserter(dest));
  }
  start_offset = (ind_to_remove.size()==0)?0:ind_to_remove[ind_to_remove.size()-1]+1;
  std::move(source.begin() + start_offset,source.end(),std::back_inserter(dest));
  source.swap(dest);
}

void FeatureTrackerBase::addFeature(const std::shared_ptr<Frame> & frame,
                                    const struct FeatureTrack & track) {
  std::size_t i = frame->num_features_++;
  frame->px_vec_.col(i) = Eigen::Vector2d((double)track.cur_pos_[0],(double)track.cur_pos_[1]);
  frame->score_vec_[i] = (double)track.first_score_;
  frame->level_vec_[i] = track.first_level_;
  frame->track_id_vec_[i] = track.track_id_;
}

void FeatureTrackerBase::addFeature(const std::shared_ptr<Frame> & frame,
                                    const int & track_id,
                                    const std::size_t & camera_id) {
  struct FeatureTrack & track = tracks_[camera_id][track_id];
  addFeature(frame,track);
}

#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
void FeatureTrackerBase::showAdditionalStat(bool include_remaining) {
  if(include_remaining) {
    /*
    * Note to future self:
    * as some tracks were not added to the statistics (the remaining tracks
    * that were NOT removed), we do some processing before
    * displaying the statistics.
    * We add all tracks with life > 0 (we ignore the newly detected features)
    */
    for(std::size_t camera_id=0;camera_id<tracks_.size();++camera_id) {
      for(const struct FeatureTrack & track : tracks_[camera_id]) {
        if(track.life_ > 0) {
          life_stat_.add(track.life_);
        }
      }
    }
  }
  life_stat_.display();
}
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */

void FeatureTrackerBase::reset(void) {
  for(std::size_t c=0;c<tracks_.size();++c) {
    tracks_[c].clear();
    tracked_features_num_[c] = 0;
    detected_features_num_[c] = 0;
  }
}

void FeatureTrackerBase::getDisparity(const double & pivot_ratio,
                                      double & total_avg_disparity) {
  /*
   * Note to future self:
   * we select the nth-element from the ordered disparities,
   * and average this for all cameras.
   * For instance, if pivot_ratio is 0.5, then we select the median
   * disparity from all cameras, and average the medians.
   */
  total_avg_disparity = 0;
  for(std::size_t c=0;c<tracks_.size();++c) {
    std::vector<float> cam_disparities;
    // only take tracks that are long enough (life >= 1)
    // break out, if we find the first with life = 0, because 
    // new tracks to to the end of the vector
    const std::vector<struct FeatureTrack> & cam_tracks = tracks_[c];
    for(std::size_t i=0;i<cam_tracks.size();++i) {
      if(cam_tracks[i].life_ > 0) {
        cam_disparities.push_back(cam_tracks[i].cur_disparity_);
      } else {
        break;
      }
    }
    if(cam_disparities.size()==0) {
      continue;
    }
    // once we have all proper dispariies, select pivot element
    const std::size_t pivot_id = pivot_ratio * cam_disparities.size();
    std::nth_element(cam_disparities.begin(),
                     cam_disparities.begin() + pivot_id,
                     cam_disparities.end());
    total_avg_disparity += cam_disparities[pivot_id];
  }
  total_avg_disparity /= tracks_.size();
}

} // namespace vilib
