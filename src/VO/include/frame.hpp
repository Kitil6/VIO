#ifndef FRAME_HPP
#define FRAME_HPP

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/video/tracking.hpp>
#include <eigen3/Eigen/Dense>
#include <list>

#define MAX_CNT      20
#define minDistance  30
#define initial_dist 20
#define MASK_R       30
#define parallax     1

class Frame{
public:
    struct points{
        std::vector<cv::Point2f>  uv;
        std::vector<cv::Point3f> xyz;
    };
    std::list<cv::Point2f> init_uv;
    // <initial_pose, <initial_uv, current_uv>> 
    std::pair<Eigen::Matrix4d, std::pair<std::list<cv::Point2f>, std::list<cv::Point2f>>> uv_new;
    // dataset_0
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 525, 0, 319.5,  0, 525, 239.5, 0, 0, 1);
    // dataset_1
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 481.20, 0, 319.50, 0, -480.00, 239.50, 0, 0, 1);
    // dataset_2
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 603.95556640625, 0, 324.0858154296875, 0, 603.1257934570312, 232.72303771972656, 0, 0, 1);
    // dataset_3
    cv::Mat K = (cv::Mat_<double>(3, 3) << 376, 0, 376, 0, 376, 240, 0, 0, 1);
    // 相机
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 479.2908854032253, 0, 296.4548676588856, 0, 479.9012226209484, 228.8420275291677, 0, 0, 1);
    points cloud, pts_last_frame, pts_curre_frame;
    Eigen::Matrix4d T_pre, T_cur;
    Eigen::Matrix4d Rt_2_T(const cv::Mat &R, const cv::Mat &t);
    Eigen::Vector4d point3f_2_vector4d(const cv::Point3f &pt);
    Eigen::Vector3d point2f_2_vector3d(const cv::Point2f &pt);
    cv::Point2f uv_2_xy(const cv::Point2f &uv);
    std::vector<cv::Point3f> check_points(const cv::Mat &Proj1, const cv::Mat &Proj2, std::vector<cv::Point2f> &xy1, std::vector<cv::Point2f> &xy2, std::vector<cv::Point2f> &uv1, std::vector<cv::Point2f> &uv2, cv::Mat &xyz, double threshold);
    bool VO_initial(cv::Mat & img1, cv::Mat & img2, int & cnt);
    void feature_tracking(cv::Mat & img1, cv::Mat & img2);
    void feature_new(cv::Mat & img1, cv::Mat & img2);
    void VO_motion_tracking(cv::Mat & img1, cv::Mat & img2);
    void point_filter_after_tracking(cv::Mat &img, std::vector<cv::Point2f> &uv1, std::vector<cv::Point2f> &uv2, int len);
    void point_filter_after_tracking(cv::Mat &img, std::vector<cv::Point2f> &uv2, std::vector<cv::Point3f> &xyz2, int len);
    double cal_parallax(Eigen::Matrix4d T_pre, Eigen::Matrix4d T_cur);
 };


#endif