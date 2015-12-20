#include "opencv2/opencv.hpp"
//#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "Card.h"

#define OPTIMIZATION_VAL 250
#define DECK_SIZE 52

using namespace cv;
using namespace std;

bool compareContours(vector<Point> a1, vector<Point> a2) {
	double a1Area = fabs(contourArea(Mat(a1)));
	double a2Area = fabs(contourArea(Mat(a2)));
	return a1Area > a2Area;
}

double distanceBetweenPoints(cv::Point2f p1, cv::Point2f p2) {
	return sqrt(pow(abs(p1.x - p2.x), 2) + pow(abs(p1.y - p2.y), 2));
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center) {
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++) {
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

vector<Card> loadDeck() {
	vector<Card> cards = vector<Card>();

	int numSuits = DECK_SIZE / 4;
	string suits[4] = { "clubs", "diamonds", "hearts", "spades" };

#pragma omp parallel for
	for (auto i = 0; i < numSuits; i++) {
		for (auto j = 0; j < 4; j++) {

			string cardType = to_string(i + 2) + "_" + suits[j];
			string cardPath = "cards\\" + cardType + ".png";
			Mat card = imread(cardPath, IMREAD_COLOR);

			if (card.empty()) {
				cout << "Error reading file " + cardType + "..." << endl;
			} else {
				cout << "Loading resource " + cardType + ".png" + "..." << endl;
				cards.push_back(Card(cardType, card));
			}
		}
	}

	return cards;
}

Card whoIsWinner(vector<Card> cards) {

	int firstIndex = rand() % 4;
	cout << "First card played: " + cards[firstIndex]._name << endl;
	size_t winner = firstIndex;

	for (size_t i = 0; i < cards.size(); i++) {
		if (i == firstIndex)
			continue;

		if (!cards[i]._suit.compare(cards[firstIndex]._suit)) {
			if (cards[i]._value > cards[firstIndex]._value) {
				winner = i;
			}
		}
	}
	return cards[winner];
}

Mat loadImageToMat(string fileName) {
	Mat srcImg = imread(fileName, IMREAD_COLOR);

	if (srcImg.empty()) {
		printf("Can't read the source image\n");
		exit(EXIT_FAILURE);
	} else {
		return srcImg;
	}
}

int main(int argc, char** argv) {
	srand((unsigned int) time(NULL));

	// Loads all the cards to the database
	vector<Card> cards = loadDeck();
	
	// Load image for analysis
	Mat srcImg = loadImageToMat("table9.jpg");

	Mat gray, blur, thresh, contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> listOfContours;

	cvtColor(srcImg, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(1, 1), 1000, 0);
	threshold(blur, thresh, 120, 255, THRESH_BINARY);

	//imshow("Display gray", gray);
	//imshow("Display blur", blur);
	//imshow("Display thresh", thresh);

	//Save copy of thresh
	contours = thresh;

	findContours(contours, listOfContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	sort(listOfContours.begin(), listOfContours.end(), compareContours);

	int numCards = 4;

	vector<Card> cardsInPlay;

	for (auto i = 0; i < numCards; i++) {
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

		for each (Point2f var in approx) {
			corners.push_back(var);
		}

		Mat homography = Mat::zeros(726, 500, CV_8UC3);
		Point2f quads[4];

		if (distanceBetweenPoints(corners[0], corners[1]) > distanceBetweenPoints(corners[1], corners[2])) {
			quads[0] = cv::Point(homography.cols, 0);
			quads[1] = cv::Point(homography.cols, homography.rows);
			quads[2] = cv::Point(0, homography.rows);
			quads[3] = cv::Point(0, 0);
		} else {
			quads[0] = cv::Point(0, 0);
			quads[1] = cv::Point(homography.cols, 0);
			quads[2] = cv::Point(homography.cols, homography.rows);
			quads[3] = cv::Point(0, homography.rows);
		}

		Point2f temp[4];
		temp[0] = corners[3];
		temp[1] = corners[2];
		temp[2] = corners[1];
		temp[3] = corners[0];

		auto transform = getPerspectiveTransform(temp, quads);
		warpPerspective(srcImg, homography, transform, homography.size());

		cardsInPlay.push_back(Card(homography));

		cv::polylines(srcImg, listOfContours[i], true, Scalar(0, 0, 255), 3);

		//DEBUG
		//namedWindow("Homography " + to_string(i + 1), 1);
		//imshow("Homography " + to_string(i + 1), homography);
	}

	FlannBasedMatcher matcher;

	for (auto k = 0; k < numCards; k++) {
		vector<DMatch> bestMatches = vector<DMatch>();
		Card matchedCard;

		for (auto j = 0; j < cards.size(); j++) {
			vector<DMatch> matches;
			vector<DMatch> goodMatches = vector<DMatch>();

			// matching descriptors
			matcher.match(cardsInPlay[k]._descriptors, cards[j]._descriptors, matches);

			for (auto i = 0; i < matches.size(); i++)
				if (matches[i].distance < OPTIMIZATION_VAL)
					goodMatches.push_back(matches[i]);

			if (bestMatches.empty()) {
				bestMatches = goodMatches;
				matchedCard = cards[j];
				continue;
			}

			if (goodMatches.size() > bestMatches.size()) {
				bestMatches = goodMatches;
				matchedCard = cards[j];
				continue;
			}
		}

		cardsInPlay[k]._name = matchedCard._name;
		cardsInPlay[k]._value = matchedCard._value;
		cardsInPlay[k]._suit = matchedCard._suit;

		// Drawing the results
		// DEBUG
		namedWindow("Matched with " + matchedCard._name, 1);
		Mat img_matches;
		drawMatches(cardsInPlay[k]._cardMatrix, cardsInPlay[k]._keyPoints,
					matchedCard._cardMatrix, matchedCard._keyPoints,
					bestMatches, img_matches);
		imshow("Matched with " + matchedCard._name, img_matches);
	}
	Card winner = whoIsWinner(cardsInPlay);

	//displayWinner(srcImg, winner);

	namedWindow("Final", 1);
	imshow("Final", srcImg);

	namedWindow("Hand Winner", 1);
	imshow("Hand Winner", winner._cardMatrix);


	waitKey(0); // Wait for a keystroke in the window
	return 0;
}