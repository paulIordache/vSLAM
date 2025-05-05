  #include <opencv2/opencv.hpp>
  #include <opencv2/features2d.hpp>
  #include <iostream>
  #include <fstream>
  #include <vector>
  #include <cmath>
  #include <algorithm>


  class RgbdVO {
   public:
    // Initialize with the camera intrinsic matrix.
    explicit RgbdVO(cv::Mat K) : K_(std::move(K)), frame_id_(0) {
      detector_ = cv::ORB::create(1000);
      current_pose_ = cv::Mat::eye(4, 4, CV_64F);
      trajectory_ = cv::Mat::zeros(600, 600, CV_8UC3);
    }

    // Process an RGB frame along with its (possibly reused) depth frame.
    bool processFrame(const cv::Mat& rgb, const cv::Mat& depth) {
      cv::Mat gray;
      cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

      // Initialization: store features from the first frame.
      if (frame_id_ == 0) {
        prev_keypoints_ = keypoints;
        prev_descriptors_ = descriptors.clone();
        prev_gray_ = gray.clone();
        prev_depth_ = depth.clone();
        frame_id_++;
        return true;
      }

      // Feature Matching.
      std::vector<cv::DMatch> matches;
      cv::BFMatcher matcher(cv::NORM_HAMMING);
      matcher.match(prev_descriptors_, descriptors, matches);

      double min_dist = 100.0;
      for (const auto& match : matches)
        min_dist = std::min(min_dist, static_cast<double>(match.distance));

      std::vector<cv::DMatch> good_matches;
      for (const auto& match : matches)
        if (match.distance < std::max(2 * min_dist, 30.0))
          good_matches.push_back(match);

      if (good_matches.size() < 10) {
        std::cout << "Not enough good matches!" << std::endl;
        return false;
      }

      // Build 3D-2D correspondences.
      std::vector<cv::Point3f> objectPoints;
      std::vector<cv::Point2f> imagePoints;
      for (const auto& match : good_matches) {
        cv::Point2f pt_prev = prev_keypoints_[match.queryIdx].pt;
        int u = static_cast<int>(pt_prev.x);
        int v = static_cast<int>(pt_prev.y);
        if (u < 0 || u >= prev_depth_.cols || v < 0 || v >= prev_depth_.rows)
          continue;
        float d = prev_depth_.at<float>(v, u);
        if (d <= 0.0f)
          continue;

        double cx = K_.at<double>(0, 2);
        double cy = K_.at<double>(1, 2);
        double fx = K_.at<double>(0, 0);
        double fy = K_.at<double>(1, 1);
        double x = (pt_prev.x - cx) * d / fx;
        double y = (pt_prev.y - cy) * d / fy;
        double z = d;
        objectPoints.emplace_back(x, y, z);

        imagePoints.push_back(keypoints[match.trainIdx].pt);
      }

      if (objectPoints.size() < 6) {
        std::cout << "Not enough valid 3D-2D correspondences!" << std::endl;
        return false;
      }

      // Pose estimation via PnP.
      cv::Mat rvec, tvec, inliers;
      bool pnp_success = cv::solvePnPRansac(objectPoints, imagePoints, K_, cv::Mat(),
                                            rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
      if (!pnp_success) {
        std::cout << "PnP failed." << std::endl;
        return false;
      }

      cv::Mat R;
      cv::Rodrigues(rvec, R);

      cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
      R.copyTo(T(cv::Rect(0, 0, 3, 3)));
      tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));

      // Update cumulative pose.
      current_pose_ = current_pose_ * T;
      trajectory_points_.push_back(current_pose_.clone());

      std::cout << "Pose (x, y, z): "
                << current_pose_.at<double>(0, 3) << ", "
                << current_pose_.at<double>(1, 3) << ", "
                << current_pose_.at<double>(2, 3) << std::endl;

      // Draw the trajectory on an image.
      int x_coord = static_cast<int>(current_pose_.at<double>(0, 3) * 10) + 300;
      int y_coord = static_cast<int>(current_pose_.at<double>(2, 3) * 10) + 300;
      cv::circle(trajectory_, cv::Point(x_coord, y_coord), 2, cv::Scalar(255, 0, 0), 2);

      cv::Mat traj_display = trajectory_.clone();
      cv::putText(traj_display, "Stereo VO (blue)", cv::Point(20, 50),
                  cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
      cv::imshow("Trajectory", traj_display);

      cv::Mat img_matches;
      cv::drawMatches(prev_gray_, prev_keypoints_, gray, keypoints, good_matches, img_matches);
      cv::imshow("Matches", img_matches);

      // Update previous frame data.
      prev_keypoints_ = keypoints;
      prev_descriptors_ = descriptors.clone();
      prev_gray_ = gray.clone();
      prev_depth_ = depth.clone();
      frame_id_++;

      return true;
    }

    void saveTrajectory(const std::string &filename) {
      std::ofstream file(filename);
      for (size_t i = 0; i < trajectory_points_.size(); i++) {
        const auto &T = trajectory_points_[i];
        file << i << " " << T.at<double>(0, 3) << " "
             << T.at<double>(1, 3) << " " << T.at<double>(2, 3) << "\n";
      }
      file.close();
    }

    // Getter for the current global pose.
    [[nodiscard]] cv::Mat getCurrentPose() const { return current_pose_; }

   private:
    cv::Mat K_;
    cv::Ptr<cv::Feature2D> detector_;
    std::vector<cv::KeyPoint> prev_keypoints_;
    cv::Mat prev_descriptors_, prev_gray_, prev_depth_;
    cv::Mat current_pose_;
    std::vector<cv::Mat> trajectory_points_;
    cv::Mat trajectory_;
    int frame_id_;
  };

  int main() {
    cv::VideoCapture cap_rgb("../rgb.mp4");
    cv::VideoCapture cap_depth("../depth.mp4");
    if (!cap_rgb.isOpened()) {
      std::cerr << "Error opening RGB video file" << std::endl;
      return -1;
    }
    if (!cap_depth.isOpened()) {
      std::cerr << "Error opening Depth video file" << std::endl;
      return -1;
    }

    double focal = 383.0, cx = 320.0, cy = 240.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, cx,
        0, focal, cy,
        0, 0, 1);
    RgbdVO vo(K);

    // Variables for timestamp-based depth alignment.
    cv::Mat lastValidDepth;
    double depthTimestamp = 0.0, rgbTimestamp = 0.0;
    const double timestampThreshold = 0.03; // 30 milliseconds

    // Data structure to store keyframe poses for local BA.
    std::vector<cv::Mat> keyframePoses;
    const int keyframeInterval = 10;
    int totalFrameCounter = 0;

    while (true) {
      cv::Mat rgb_frame;
      cap_rgb >> rgb_frame;
      if (rgb_frame.empty())
        break;

      // Retrieve the current RGB timestamp.
      rgbTimestamp = cap_rgb.get(cv::CAP_PROP_POS_MSEC) / 1000.0;

      cv::Mat depth_frame;
      // Attempt to grab a depth frame.
      bool depthFrameRead = cap_depth.grab();
      if (depthFrameRead) {
        cap_depth.retrieve(depth_frame);
        depthTimestamp = cap_depth.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        if (std::abs(depthTimestamp - rgbTimestamp) < timestampThreshold) {
          if (depth_frame.type() != CV_32F)
            depth_frame.convertTo(depth_frame, CV_32F);
          lastValidDepth = depth_frame.clone();
        } else {
          depth_frame = lastValidDepth;
        }
      } else {
        depth_frame = lastValidDepth;
      }

      if (depth_frame.empty()) {
        std::cerr << "No valid depth frame available for timestamp " << rgbTimestamp << std::endl;
        continue;
      }

      if (vo.processFrame(rgb_frame, depth_frame)) {
        totalFrameCounter++;

        // Save keyframes periodically.
        if (totalFrameCounter % keyframeInterval == 0) {
          keyframePoses.push_back(vo.getCurrentPose().clone());
        }

      }

      if (cv::waitKey(1) == 27)
        break;
    }

    vo.saveTrajectory("stereo_trajectory.txt");
    return 0;
  }
