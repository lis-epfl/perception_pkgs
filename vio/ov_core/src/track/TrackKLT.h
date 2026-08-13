/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"

namespace ov_core {

/**
 * @brief KLT tracking of features.
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp)
 * OpenCV function to do the KLT tracking.
 */
class TrackKLT : public TrackBase {

public:
  bool use_polar_grid = false;
  int polar_rings = 4, polar_sectors = 8;
  std::map<size_t, std::pair<cv::Point2f, float>> polar_geom; // cam -> (circle centre, radius)

  /// (ring, sector) bin index for a point, or -1 if outside the image circle.
  /// Rings carry EQUAL SOLID ANGLE under the equidistant model r = f*theta: the k-th boundary
  /// is at theta_k = acos(1 - (k/K)(1 - cos theta_max)), i.e. r_k = R * theta_k / theta_max.
  /// Equal-area-in-image rings would over-populate the periphery in bearing terms.
  int polar_bin(size_t cam_id, const cv::Point2f &pt, const cv::Size &sz, const cv::Mat &mask) {
    auto it = polar_geom.find(cam_id);
    if (it == polar_geom.end()) {
      // Derive the circle from the mask: centre = centroid of valid pixels, R = max radius.
      double sx = 0, sy = 0; long n = 0;
      for (int y = 0; y < mask.rows; y += 4)
        for (int x = 0; x < mask.cols; x += 4)
          if (mask.at<uint8_t>(y, x) <= 127) { sx += x; sy += y; n++; }
      cv::Point2f c = n ? cv::Point2f((float)(sx / n), (float)(sy / n))
                        : cv::Point2f(sz.width / 2.0f, sz.height / 2.0f);
      float R = 1.0f;
      for (int y = 0; y < mask.rows; y += 4)
        for (int x = 0; x < mask.cols; x += 4)
          if (mask.at<uint8_t>(y, x) <= 127)
            R = std::max(R, (float)std::hypot(x - c.x, y - c.y));
      polar_geom[cam_id] = {c, R};
      it = polar_geom.find(cam_id);
    }
    const cv::Point2f &c = it->second.first;
    float R = it->second.second;
    float dx = pt.x - c.x, dy = pt.y - c.y;
    float r = std::sqrt(dx * dx + dy * dy);
    if (r > R) return -1;
    const double th_max = 1.83;                 // platform lens half-FOV (rad)
    double th = th_max * (double)(r / R);
    double frac = (1.0 - std::cos(th)) / (1.0 - std::cos(th_max));   // solid-angle fraction
    int ring = (int)(frac * polar_rings);
    if (ring >= polar_rings) ring = polar_rings - 1;
    if (ring < 0) ring = 0;
    double phi = std::atan2((double)dy, (double)dx) + M_PI;          // [0, 2pi)
    int sec = (int)(phi / (2.0 * M_PI) * polar_sectors);
    if (sec >= polar_sectors) sec = polar_sectors - 1;
    if (sec < 0) sec = 0;
    return ring * polar_sectors + sec;
  }

  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   */
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist) {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Detects new features in the current image
   * @param img0pyr image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param ids0 vector of feature ids for each currently extracted keypoint
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0, size_t cam_id = 0);

  /**
   * @brief Detects new features in the current stereo pair
   * @param img0pyr left image we will detect features on (first level of pyramid)
   * @param img1pyr right image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param mask1 mask which has what ROI we do not want features in
   * @param cam_id_left first camera sensor id
   * @param cam_id_right second camera sensor id
   * @param pts0 left vector of currently extracted keypoints
   * @param pts1 right vector of currently extracted keypoints
   * @param ids0 left vector of feature ids for each currently extracted keypoint
   * @param ids1 right vector of feature ids for each currently extracted keypoint
   *
   * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
   * So we detect features in the left image, and then KLT track them onto the right image.
   * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   */
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);


  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
