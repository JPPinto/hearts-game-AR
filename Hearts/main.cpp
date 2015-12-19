#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <Windows.h>

#define OPTIMIZATION_VAL 200
#define DECK_SIZE 52

using namespace cv;
using namespace std;

bool compareContours(vector<Point> a1, vector<Point> a2){
	double a1Area = fabs(contourArea(Mat(a1)));
	double a2Area = fabs(contourArea(Mat(a2)));
	return a1Area > a2Area;
}

double distanceBetweenPoints(cv::Point2f p1, cv::Point2f p2){
	return sqrt(pow(abs(p1.x - p2.x), 2) + pow(abs(p1.y - p2.y), 2));
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

vector<Mat> loadDeck(){
	vector<Mat> cards = vector<Mat>();
	int numSuits = DECK_SIZE / 4;
	String suits[4] = { "clubs", "diamonds", "hearts", "spades" };

	for (auto i = 0; i < numSuits; i++){
		for (auto j = 0; j < 4; j++){

			String cardPath = "cards\\" + to_string(i + 2) + "_" + suits[j] + ".png";
			Mat card = imread(cardPath, IMREAD_COLOR);

			if (card.empty())
				cout << "Error reading file..." << endl;
			else{
				cout << "Loading resource " + to_string(i + 2) + "_" + suits[j] + ".png" + "..." << endl;
				cards.push_back(card);
			}
		}
	}

	return cards;
}

int main(int argc, char** argv)
{
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.

	vector<Mat> cards = vector<Mat>();

	cards = loadDeck();

	Mat srcImg = imread("example_4.png", IMREAD_COLOR);

	if (srcImg.empty())
	{
		printf("Can't read the source image\n");
		getchar();
		return -1;
	}

	Mat gray, blur, thresh, contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> listOfContours;
	cvtColor(srcImg, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(1, 1), 1000, 0);
	threshold(blur, thresh, 120, 255, THRESH_BINARY);

	imshow("Display gray", gray);
	imshow("Display blur", blur);
	imshow("Display thresh", thresh);

	//Save copy of thresh
	contours = thresh;

	findContours(contours, listOfContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	sort(listOfContours.begin(), listOfContours.end(), compareContours);

	int numCards = 1;

	Mat homography;

	for (auto i = 0; i < numCards; i++)
	{
		auto card = listOfContours[i];
		auto peri = arcLength(card, true);
		vector<Point> approx;
		approxPolyDP(card, approx, 0.02*peri, true);



		auto rect = minAreaRect(listOfContours[i]);
		CvPoint2D32f r[4];
		cvBoxPoints(rect, r);
		vector<Point2f> rectangle;
		for each (CvPoint2D32f var in r)
			rectangle.push_back(var);

		vector<Point2f> corners = vector<Point2f>();

		for each (Point2f var in approx)
		{
			corners.push_back(var);
		}

		homography = Mat::zeros(726, 500, CV_8UC3);
		Point2f quads[4];

		if (distanceBetweenPoints(corners[0], corners[1]) > distanceBetweenPoints(corners[1], corners[2])){
			quads[0] = cv::Point((float)homography.cols, (float)0);
			quads[1] = cv::Point((float)homography.cols, (float)homography.rows);
			quads[2] = cv::Point((float)0, (float)homography.rows);
			quads[3] = cv::Point((float)0, (float)0);
		}
		else {
			quads[0] = cv::Point((float)0, (float)0);
			quads[1] = cv::Point((float)homography.cols, (float)0);
			quads[2] = cv::Point((float)homography.cols, (float)homography.rows);
			quads[3] = cv::Point((float)0, (float)homography.rows);
		}

		sortCorners(corners, rect.center);

		Point2f temp[4];
		temp[0] = corners[0];
		temp[1] = corners[1];
		temp[2] = corners[2];
		temp[3] = corners[3];

		auto transform = getPerspectiveTransform(temp, quads);
		warpPerspective(srcImg, homography, transform, homography.size());

		imshow("Homography", homography);
	}

	// detecting keypoints
	SiftFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;

	// computing descriptors
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;

	// matching descriptors
	FlannBasedMatcher matcher;

	vector<DMatch> bestMatches = vector<DMatch>();
	Mat bestMatchImg;
	vector<KeyPoint> bestKeypoints;

	//for (auto k = 0; k < numCards; k++)
	//{
	detector.detect(homography, keypoints1);
	extractor.compute(homography, keypoints1, descriptors1);


	for (auto j = 0; j < cards.size(); j++)
	{
		detector.detect(cards[j], keypoints2);
		extractor.compute(cards[j], keypoints2, descriptors2);

		vector<DMatch> matches;
		vector<DMatch> goodMatches = vector<DMatch>();

		matcher.match(descriptors1, descriptors2, matches);

		for (auto i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance < OPTIMIZATION_VAL){
				goodMatches.push_back(matches[i]);
			}
		}

		if (bestMatches.empty()){
			bestMatches = goodMatches;
			bestKeypoints = keypoints2;
			bestMatchImg = cards[j];
			continue;
		}

		if (goodMatches.size() > bestMatches.size()){
			bestMatches = goodMatches;
			bestKeypoints = keypoints2;
			bestMatchImg = cards[j];
			continue;
		}
	}
	//}

	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(homography, keypoints1, bestMatchImg, bestKeypoints, bestMatches, img_matches);
	imshow("matches", img_matches);


	waitKey(0); // Wait for a keystroke in the window
	return 0;
}