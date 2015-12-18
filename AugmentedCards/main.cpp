#include "opencv2/opencv.hpp"
#include <iostream>

#include <math.h>
#include <windows.h>

using namespace cv;

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
	int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d;
		pt.y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}

	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

int main()
{
	Mat dst, color_dst;
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Source", 1);
	namedWindow("Detected Lines", 1);

	for (;;){
		Mat src;
		cap >> src;

		Canny(src, dst, 200, 200, 3);
		cvtColor(dst, color_dst, CV_GRAY2BGR);

		std::vector<cv::Vec4i> lines;
		HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);

		std::vector<cv::Point2f> corners;
		for (int i = 0; i < lines.size(); i++)
		{
			for (int j = i + 1; j < lines.size(); j++)
			{
				cv::Point2f pt = computeIntersect(lines[i], lines[j]);
				if (pt.x >= 0 && pt.y >= 0){
					corners.push_back(pt);
					std::cout << "x: " << pt.x << " y: " << pt.y << std::endl;
				}
			}
		}

		std::vector<cv::Point2f> approx;
		cv::approxPolyDP(cv::Mat(corners), approx,
			cv::arcLength(cv::Mat(corners), true) * 0.02, true);

		if (approx.size() != 4){
			std::cout << "The object is not quadrilateral!" << std::endl;
		}
		else if (corners.size() >= 4){
			std::cout << "The object is quadrilateral!" << std::endl;

			// Get mass center
			cv::Point2f center(0, 0);
			for (int i = 0; i < corners.size(); i++)
				center += corners[i];

			center *= (1. / corners.size());
			sortCorners(corners, center);

			// Define the destination image
			cv::Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);

			// Corners of the destination image
			std::vector<cv::Point2f> quad_pts;
			quad_pts.push_back(cv::Point2f(0, 0));
			quad_pts.push_back(cv::Point2f(quad.cols, 0));
			quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
			quad_pts.push_back(cv::Point2f(0, quad.rows));

			// Get transformation matrix
			cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);

			// Apply perspective transformation
			cv::warpPerspective(src, quad, transmtx, quad.size());
			cv::imshow("quadrilateral", quad);
		}

		//drawlines
		for (size_t i = 0; i < lines.size(); i++){
			line(src, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
		}

		imshow("Source", src);
		imshow("Detected Lines", color_dst);

		if (waitKey(30) >= 0)
			break;

		Sleep(1000);

	}
	return 0;
}