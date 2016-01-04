/* General includes */
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>

/* OpenCV includes */
#include "opencv2/opencv.hpp"
//#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>

/* OpenMP includes */
#include <omp.h>

/* This app includes */
#include "Card.h"

/* Debug defines */
//#define DEBUG_INITIAL_TRANSFORMS
//#define DEBUG_HOMOGRAPHY
//#define DEBUG_CARD_MATCHES

#define OPTIMIZATION_VAL 200
#define DECK_SIZE 52

/* Gaussian blur parameters */
#define GAUSSIAN_BLUR_SIZE_X 5
#define GAUSSIAN_BLUR_SIZE_Y 5
#define GAUSSIAN_BLUR_SIGMA_X 1000
#define GAUSSIAN_BLUR_SIGMA_Y 1000

/* Threshold parameters */
#define THRESHOLD_MIN 0
#define THRESHOLD_MAX 255

/* Threshold merge parameters */
#define THRESHOLD_MERGE_MIN 0
#define THRESHOLD_MERGE_MAX 255

/* Colors */
#define CONTOUR_COLOR			Scalar(0,     0, 255)
#define WINNER_CONTOUR_COLOR	Scalar(26,  255,   0)
#define LOSER_FONT_COLOR		Scalar(0,   195, 255)
#define WINNER_FONT_COLOR		Scalar(0,   255,  36)

using namespace cv;
using namespace std;

bool compareContours(vector<Point> a1, vector<Point> a2) {
	double a1Area = fabs(contourArea(Mat(a1)));
	double a2Area = fabs(contourArea(Mat(a2)));
	return a1Area > a2Area;
}

double distanceBetweenPoints(Point2f p1, Point2f p2) {
	return sqrt(pow(abs(p1.x - p2.x), 2) + pow(abs(p1.y - p2.y), 2));
}

vector<Card> loadDeck() {
	cout << "Loading card images: ";
	unsigned int cardsLoaded = 0;
	vector<Card> cards = vector<Card>();

	int numSuits = DECK_SIZE / 4;
	string suits[4] = { "clubs", "diamonds", "hearts", "spades" };

	for (int i = 0; i < numSuits; i++) {
		for (int j = 0; j < 4; j++) {

			string cardType = to_string(i + 2) + "_" + suits[j];
			string cardPath = "cards\\" + cardType + ".png";
			Mat card = imread(cardPath, IMREAD_COLOR);

			if (card.empty()) {
				cout << "Error reading file " + cardType + "..." << endl;
			} else {
				Card temp = Card(cardType, card);
				cards.push_back(temp);
				cardsLoaded++;
			}
		}
	}
	cout << cardsLoaded << " cards loaded." << endl;

	cout << "Performing sift using " << omp_get_max_threads() << " threads: ";

#pragma omp parallel for
	for (int i = 0; i < cards.size(); i++) {
		cards[i].doSift();
	}

	cout << " sift done." << endl;
	return cards;
}

Mat loadImageToMat(string fileName) {
	cout << "Loading image for analysis: ";
	Mat srcImg = imread(fileName, IMREAD_COLOR);

	if (srcImg.empty()) {
		cout << "Can't read the source image. Aborting." << endl;
		getchar();
		exit(EXIT_FAILURE);
	} else {
		cout << "Image loaded." << endl;
		return srcImg;
	}
}

Mat mergeImages(Mat img1, Mat img2) {

	Mat gray, gray_inv, tempFinal1, tempFinal2;

	cvtColor(img2, gray, CV_BGR2GRAY);
	threshold(gray, gray, THRESHOLD_MERGE_MIN, THRESHOLD_MERGE_MAX, CV_THRESH_BINARY);

	bitwise_not(gray, gray_inv);

	img1.copyTo(tempFinal1, gray_inv);
	img2.copyTo(tempFinal2, gray);

	Mat final;
	final = tempFinal1 + tempFinal2;
	return final;
}

int main(int argc, char** argv) {
	srand((unsigned int)time(NULL));

	string srcImgPath;

	if (argc < 2 || argc > 2) {
		cout << "Image for detection was not provided. Using default." << endl;
		srcImgPath = "table1.png";
	} else {
		srcImgPath = argv[1];
	}

	/* Load image for analysis */
	Mat srcImg = loadImageToMat(srcImgPath);

	/* Loads all the cards to the database */
	vector<Card> cards = loadDeck();

	Mat grayScaleMat, gaussianBlurMat, thresholdMat;

	/* Convert to grayscales */
	cvtColor(srcImg, grayScaleMat, COLOR_BGR2GRAY);
	/* Gaussian blur */
	GaussianBlur(grayScaleMat, gaussianBlurMat, Size(GAUSSIAN_BLUR_SIZE_X, GAUSSIAN_BLUR_SIZE_Y), GAUSSIAN_BLUR_SIGMA_X, GAUSSIAN_BLUR_SIGMA_Y, 0);
	/* Apply threshold */
	threshold(gaussianBlurMat, thresholdMat, THRESHOLD_MIN, THRESHOLD_MAX, THRESH_BINARY);

#ifdef DEBUG_INITIAL_TRANSFORMS
	imshow("Display gray", grayScaleMat);
	imshow("Display blur", gaussianBlurMat);
	imshow("Display thresh", thresholdMat);
#endif

	/* Save copy of thresh */
	Mat contoursMat = thresholdMat;

	vector<Vec4i> hierarchy;
	vector<vector<Point>> listOfContours;

	findContours(contoursMat, listOfContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	sort(listOfContours.begin(), listOfContours.end(), compareContours);

	int numCardsOnDisplay = 4;

	vector<Card> cardsInPlay;

	for (auto i = 0; i < numCardsOnDisplay; i++) {
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
			quads[0] = Point(homography.cols, 0);
			quads[1] = Point(homography.cols, homography.rows);
			quads[2] = Point(0, homography.rows);
			quads[3] = Point(0, 0);
		} else {
			quads[0] = Point(0, 0);
			quads[1] = Point(homography.cols, 0);
			quads[2] = Point(homography.cols, homography.rows);
			quads[3] = Point(0, homography.rows);
		}

		Point2f temp[4];
		temp[0] = corners[3];
		temp[1] = corners[2];
		temp[2] = corners[1];
		temp[3] = corners[0];

		auto transform = getPerspectiveTransform(temp, quads);
		warpPerspective(srcImg, homography, transform, homography.size());

		vector<Point2f> srcPoints;
		vector<Point2f> destPoints;

		for (size_t j = 0; j < 4; j++) {
			srcPoints.push_back(quads[j]);
		}

		for (size_t j = 0; j < 4; j++) {
			destPoints.push_back(temp[j]);
		}

		/* Draw of contours along the 4 cards */
		polylines(srcImg, listOfContours[i], true, CONTOUR_COLOR, 3);

		/* Creation and wraping of text homographies */

		/* Create empty matrixs for each case */
		int baseline = 0;
		Mat loserTextMatrix = Mat::zeros(726, 500, CV_8UC3);
		Mat winnerTextMatrix = Mat::zeros(726, 500, CV_8UC3);

		/* Homography of the card with only the text "Loser" */
		Size textSizeLoser = getTextSize("Loser", FONT_HERSHEY_SCRIPT_SIMPLEX, 5, 4, &baseline);
		Point loserTextOrg((loserTextMatrix.cols - textSizeLoser.width) / 2, (loserTextMatrix.rows + textSizeLoser.height) / 2);
		putText(loserTextMatrix, "Loser", loserTextOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, 5, LOSER_FONT_COLOR, 4, CV_AA);

		/* Homography of the card with only the text "Winner" */
		Size textSizeWinner = getTextSize("Winner", FONT_HERSHEY_SCRIPT_SIMPLEX, 5, 4, &baseline);
		Point winnerTextOrg((winnerTextMatrix.cols - textSizeWinner.width) / 2, (winnerTextMatrix.rows + textSizeWinner.height) / 2);
		putText(winnerTextMatrix, "Winner", winnerTextOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, 5, WINNER_FONT_COLOR, 4, CV_AA);

		/* Transform matrix that was applied to the card to obtain the homograpy */
		Mat textHomography = findHomography(srcPoints, destPoints);

		Mat loserTextWarped;
		Mat winnerTextWarped;
		warpPerspective(loserTextMatrix, loserTextWarped, textHomography, srcImg.size());
		warpPerspective(winnerTextMatrix, winnerTextWarped, textHomography, srcImg.size());

		cardsInPlay.push_back(Card(homography, winnerTextWarped, loserTextWarped, listOfContours[i]));

#pragma omp parallel for
		for (int m = 0; m < cardsInPlay.size(); m++) {
			cardsInPlay[m].doSift();
		}

#ifdef DEBUG_HOMOGRAPHY
		namedWindow("Homography " + to_string(i + 1), 1);
		imshow("Homography " + to_string(i + 1), homography);
#endif
	}

	cout << "Matching cards: ";

	FlannBasedMatcher matcher;

	for (auto k = 0; k < numCardsOnDisplay; k++) {
		vector<DMatch> bestMatches = vector<DMatch>();
		Card matchedCard;

		/* Compare obtained cards with cards in database  */
#pragma omp parallel for
		for (int j = 0; j < cards.size(); j++) {
			vector<DMatch> matches;
			vector<DMatch> goodMatches = vector<DMatch>();

			/* matching descriptors (matches -> output)  */
			matcher.match(cardsInPlay[k].getDescriptors(), cards[j].getDescriptors(), matches);

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
				} else if (goodMatches.size() > bestMatches.size()) {
					bestMatches = goodMatches;
					matchedCard = cards[j];
				}
			}
		}

		cardsInPlay[k].setName(matchedCard.getName());
		cardsInPlay[k].setValue(matchedCard.getValue());
		cardsInPlay[k].setSuit(matchedCard.getSuit());

#ifdef DEBUG_CARD_MATCHES
		/* Drawing the results  */
		namedWindow("Matched with " + matchedCard.getName(), 1);
		Mat img_matches;
		drawMatches(cardsInPlay[k].getCardMatrix(), cardsInPlay[k].getKeyPoints(),
					matchedCard.getCardMatrix(), matchedCard.getKeyPoints(),
					bestMatches, img_matches);
		imshow("Matched with " + matchedCard.getName(), img_matches);
#endif
	}

	cout << "Cards matched." << endl;

	Card winner = Card::whoIsWinner(cardsInPlay);

	Mat finalImg = srcImg;

	polylines(finalImg, winner.getContours(), true, WINNER_CONTOUR_COLOR, 3);

	for (size_t i = 0; i < cardsInPlay.size(); i++) {
		if (cardsInPlay[i].getName() != winner.getName()) {
			finalImg = mergeImages(finalImg, cardsInPlay[i].getLoserHomography());
		} else {
			finalImg = mergeImages(finalImg, winner.getWinnerHomography());
		}
	}

	namedWindow("Final", 1);
	imshow("Final", finalImg);

	waitKey(0); /* Wait for a keystroke in the window */
	return 0;
}