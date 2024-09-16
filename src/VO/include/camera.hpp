#ifndef CAMERA_HPP
#define CAMERA_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <eigen3/Eigen/Dense>
#include <frame.hpp>

class Camera{
public:
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
    void get_img(cv::Mat & img, cv::VideoCapture &cap);
    cv::Point2f pixel_2_norm(const cv::Point2f &pt);
    static void set_mask(const Frame::points &pts2, cv::Mat &img, int r)
    {
        cv::Mat mask = cv::Mat::ones(img.size(), CV_8UC1);
        int col = img.cols;
        int row = img.rows;
        int len = r * 2;
        float a, b, c, d;
        for (auto i : pts2.uv)
        {
            a = i.x - r;
            b = i.y - r;
            c = d = len;
            if (a < 0)
                a = 0;
            if (b < 0)
                b = 0;
            if (a + len > col)
                c = col - a - 1;
            if (b + len > row)
                d = row - b - 1;
            cv::Rect roi(a, b, c, d);
            mask(roi).setTo(cv::Scalar(0));
        }
        img = mask.clone();
    };

    static float calc_dist(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2)
    {
        float x_sum = 0, y_sum = 0;
        std::size_t size = pts1.size();

        for (int count = 0; count < size; count++)
        {
            x_sum += abs(pts1[count].x - pts2[count].x);
            y_sum += abs(pts1[count].y - pts2[count].y);
        }
        // printf("\n x = %lf, y = %lf", x_sum/size, y_sum/size);
        return x_sum / size + y_sum / size;
    }
};



#endif