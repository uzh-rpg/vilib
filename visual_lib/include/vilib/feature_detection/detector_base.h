/*
 * Base class for feature detectors
 * detector_base.h
 */

#pragma once

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include "vilib/common/occupancy_grid_2d.h"
#include "vilib/common/frame.h"

namespace vilib {

class DetectorBase {
public:
  // identified interest points
  struct FeaturePoint {
    double x_;
    double y_;
    double score_;
    unsigned int level_;

    FeaturePoint(double x, double y, double score, int level) :
      x_(x),y_(y),score_(score),level_(level) {}
    ~FeaturePoint(void) = default;
  };

  DetectorBase(const std::size_t image_width,
               const std::size_t image_height,
               const std::size_t cell_size_width,
               const std::size_t cell_size_height,
               const std::size_t min_level,
               const std::size_t max_level,
               const std::size_t horizontal_border,
               const std::size_t vertical_border);
  virtual ~DetectorBase(void) = default;

  inline OccupancyGrid2D & getGrid(void) { return grid_; }
  inline const std::vector<struct FeaturePoint> & getPoints(void) const { return keypoints_; }
  inline const std::size_t & getCellSizeWidth(void) { return cell_size_width_; }
  inline const std::size_t & getCellSizeHeight(void) { return cell_size_height_; }
  inline const std::size_t & getCellCountHorizontal(void) { return n_cols_; }
  inline const std::size_t & getCellCountVertical(void) { return n_rows_; }

  virtual void reset(void);
  virtual void detect(const std::vector<cv::Mat> & image);
  void addFeaturePoint(double x, double y, double score, unsigned int level);
  virtual std::size_t count(void) const;

  void displayFeatureGrid(const char * title,
                          const std::vector<cv::Mat> & image_pyramid,
                          bool draw_on_level0,
                          bool draw_cells) const;
  void displayFeatures(const char * title,
                       const std::vector<cv::Mat> & image_pyramid,
                       bool draw_on_level0) const;
protected:
  // cell size (width & height)
  std::size_t cell_size_width_;
  std::size_t cell_size_height_;
  // number of cells in the horizontal direction
  std::size_t n_cols_;
  // number of rows in the vertical direction
  std::size_t n_rows_;
  // minimum pyramid level to execute the feature detection on
  std::size_t min_level_;
  // maximum pyramid level TO execute the feature detection on
  std::size_t max_level_;
  // borders (applicable on both sides)
  std::size_t horizontal_border_;
  std::size_t vertical_border_;
  // identified keypoints (n_cols_ x n_rows_)
  std::vector<struct FeaturePoint> keypoints_;
  // occupancy grid
  OccupancyGrid2D grid_;
};

} // namespace vilib
