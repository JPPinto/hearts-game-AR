#include "opencv2/opencv.hpp"
//#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "Card.h"

// Debug defines
// #define DEBUG_INITIAL_TRANSFORMS
// #define DEBUG_HOMOGRAPHY
// #define DEBUG_CARD_MATCHES

#define OPTIMIZATION_VAL 250
#define DECK_SIZE 52

#define GAUSSIAN_BLUR_SIZE_X 1
#define GAUSSIAN_BLUR_SIZE_Y 1
#define GAUSSIAN_BLUR_SIGMA_X 1000
#define GAUSSIAN_BLUR_SIGMA_Y 1000

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

vector<Card> loadDeck() {
	cout << "Loading resources using " << omp_get_max_threads() << " cores." << endl;
	unsigned int cardsLoaded = 0;
	vector<Card> cards = vector<Card>();

	int numSuits = DECK_SIZE / 4;
	string suits[4] = { "clubs", "diamonds", "hearts", "spades" };

#pragma omp parallel for
	for (int i = 0; i < numSuits; i++) {
		for (int j = 0; j < 4; j++) {

			string cardType = to_string(i + 2) + "_" + suits[j];
			string cardPath = "cards\\" + cardType + ".png";
			Mat card = imread(cardPath, IMREAD_COLOR);

			if (card.empty()) {
				cout << "Error reading file " + cardType + "..." << endl;
			}
			else {
				cout << "Loading resource " + cardType + ".png" + "..." << endl;
				Card temp = Card(cardType, card);
#pragma omp critical
				{
					cards.push_back(temp);
					cardsLoaded++;
				}
			}
		}
	}

	cout << "Resources loaded. " << cardsLoaded << " cards loaded." << endl;
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
	cout << "Loading image for analysis..." << endl;
	Mat srcImg = imread(fileName, IMREAD_COLOR);

	if (srcImg.empty()) {
		cout << "Can't read the source image. Aborting." << endl;
		exit(EXIT_FAILURE);
	}
	else {
		return srcImg;
	}
}

Mat mergeImages(Mat img1, Mat img2){

	Mat gray, gray_inv, tempFinal1, tempFinal2;

	cvtColor(img2, gray, CV_BGR2GRAY);
	threshold(gray, gray, 0, 255, CV_THRESH_BINARY);

	bitwise_not(gray, gray_inv);

	img1.copyTo(tempFinal1, gray_inv);
	img2.copyTo(tempFinal2, gray);

	Mat final;
	final = tempFinal1 + tempFinal2;
	return final;
}

int main(int argc, char** argv) {
	srand((unsigned int)time(NULL));

	// Load image for analysis
	Mat srcImg = loadImageToMat("table1.png");

	// Loads all the cards to the database
	vector<Card> cards = loadDeck();

	Mat grayScaleMat, gaussianBlurMat, thresholdMat, contoursMat;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> listOfContours;

	// Convert to grayscales
	cvtColor(srcImg, grayScaleMat, COLOR_BGR2GRAY);
	// Gaussian blur
	GaussianBlur(grayScaleMat, gaussianBlurMat, Size(GAUSSIAN_BLUR_SIZE_X, GAUSSIAN_BLUR_SIZE_Y), GAUSSIAN_BLUR_SIGMA_X, GAUSSIAN_BLUR_SIGMA_Y, 0);
	// Apply thresold
	threshold(gaussianBlurMat, thresholdMat, 120, 255, THRESH_BINARY);

#ifdef DEBUG_INITIAL_TRANSFORMS
	imshow("Display gray", gray);
	imshow("Display blur", blur);
	imshow("Display thresh", thresh);
#endif

	//Save copy of thresh
	contoursMat = thresholdMat;

	findContours(contoursMat, listOfContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	sort(listOfContours.begin(), listOfContours.end(), compareContours);

	int numCards = 4;

	vector<Card> cardsInPlay;

	for (auto i = 0; i < numCards; i++) {
		auto card = listOfContours[i];
		auto peri = arcLength(card, true);
		vector<Point> approx;
		approxPolyDP(card, approx, 0.02*peri, true);

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

		cv::polylines(srcImg, listOfContours[i], true, Scalar(0, 0, 255), 3);

		/* Creation and wraping of text homographies */

		// Create empty matrixs for each case
		int baseline = 0;
		Mat loserTextMatrix = Mat::zeros(726, 500, CV_8UC3);
		Mat winnerTextMatrix = Mat::zeros(726, 500, CV_8UC3);

		//Homography of the card with only the text "Loser"
		Size textSizeLoser = getTextSize("Loser", FONT_HERSHEY_SCRIPT_SIMPLEX, 5, 4, &baseline);
		Point loserTextOrg((loserTextMatrix.cols - textSizeLoser.width) / 2, (loserTextMatrix.rows + textSizeLoser.height) / 2);
		putText(loserTextMatrix, "Loser", loserTextOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, 5, Scalar(0, 195, 255), 4, CV_AA);

		//Homography of the card with only the text "Winner"
		Size textSizeWinner = getTextSize("Winner", FONT_HERSHEY_SCRIPT_SIMPLEX, 5, 4, &baseline);
		Point winnerTextOrg((winnerTextMatrix.cols - textSizeWinner.width) / 2, (winnerTextMatrix.rows + textSizeWinner.height) / 2);
		putText(winnerTextMatrix, "Winner", winnerTextOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, 5, Scalar(0, 255, 36), 4, CV_AA);

		vector<Point2f> srcPoints;
		vector<Point2f> destPoints;

		for (size_t i = 0; i < 4; i++) {
			srcPoints.push_back(quads[i]);
		}
			
		for (size_t i = 0; i < 4; i++) {
			destPoints.push_back(temp[i]);
		}

		//Transform matrix that was applied to the card to obtain the homograpy
		Mat textHomography = findHomography(srcPoints, destPoints);

		Mat loserTextWarped;
		Mat winnerTextWarped;
		warpPerspective(loserTextMatrix, loserTextWarped, textHomography, srcImg.size());
		warpPerspective(winnerTextMatrix, winnerTextWarped, textHomography, srcImg.size());

		cardsInPlay.push_back(Card(homography, winnerTextWarped, loserTextWarped));

#ifdef DEBUG_HOMOGRAPHY
		namedWindow("Homography " + to_string(i + 1), 1);
		imshow("Homography " + to_string(i + 1), homography);
#endif
	}

	cout << "Matching cards" << endl;

	FlannBasedMatcher matcher;

	for (auto k = 0; k < numCards; k++) {
		vector<DMatch> bestMatches = vector<DMatch>();
		Card matchedCard;

		// Compare obtained cards with cards in database
#pragma omp parallel for
		for (int j = 0; j < cards.size(); j++) {
			vector<DMatch> matches;
			vector<DMatch> goodMatches = vector<DMatch>();

			// matching descriptors (matches -> output)
			matcher.match(cardsInPlay[k]._descriptors, cards[j]._descriptors, matches);

			for (int i = 0; i < matches.size(); i++) {
				if (matches[i].distance < OPTIMIZATION_VAL) {
					goodMatches.push_back(matches[i]);
				}
			}

#pragma omp critical
			{
				if (bestMatches.empty()) {
					bestMatches = goodMatches;
					matchedCard = cards[j];
				}
				else if (goodMatches.size() > bestMatches.size()) {
					bestMatches = goodMatches;
					matchedCard = cards[j];
				}
			}
		}

		cardsInPlay[k]._name = matchedCard._name;
		cardsInPlay[k]._value = matchedCard._value;
		cardsInPlay[k]._suit = matchedCard._suit;

		// Drawing the results
#ifdef DEBUG_CARD_MATCHES
		namedWindow("Matched with " + matchedCard._name, 1);
		Mat img_matches;
		drawMatches(cardsInPlay[k]._cardMatrix, cardsInPlay[k]._keyPoints,
			matchedCard._cardMatrix, matchedCard._keyPoints,
			bestMatches, img_matches);
		imshow("Matched with " + matchedCard._name, img_matches);
#endif
	}

	Card winner = whoIsWinner(cardsInPlay);

	Mat finalImg = srcImg;

	for (size_t i = 0; i < cardsInPlay.size(); i++) {
		if (cardsInPlay[i]._name != winner._name) {
			finalImg = mergeImages(finalImg, cardsInPlay[i]._loserHomography);
		} else {
			finalImg = mergeImages(finalImg, winner._winnerHomography);
		}
	}

	namedWindow("Final", 1);
	imshow("Final", finalImg);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}