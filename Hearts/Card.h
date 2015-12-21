#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <string>

#define SIFT_DETECTOR_PARAMETER 400

using namespace cv;
using namespace std;

class Card {
private:
	string _name;
	string _suit;
	int _value;

	Mat _winnerHomography;
	Mat _loserHomography;

	Mat _cardMatrix;
	vector<KeyPoint> _keyPoints;
	Mat _descriptors;

public:
	Card() {}
	Card(Mat mat, Mat winnerHomography, Mat loserHomography);
	Card(std::string name, Mat mat);
	~Card();

	void doSift();
	
	/* Setters and getters */
	void setName(string newName);
	void setSuit(string newSuit);
	void setValue(int newValue);

	string getName();
	string getSuit();
	int getValue();

	Mat getWinnerHomography();
	Mat getLoserHomography();
	Mat getDescriptors();

	/* Static methods */
	static Card whoIsWinner(vector<Card> cards);
};

Card::Card(Mat mat, Mat winnerHomography, Mat loserHomography) {
	_name = "Unknown";
	_cardMatrix = mat;

	_winnerHomography = winnerHomography;
	_loserHomography = loserHomography;
}

Card::Card(std::string name, Mat mat) {
	_name = name;
	_cardMatrix = mat;

	size_t underscore = _name.find_first_of('_');

	_value = atoi(_name.substr(0, underscore).c_str());
	_suit = _name.substr(underscore + 1);
}

void Card::doSift() {
	/* Calculade this card's keypoints and descriptors */
	SiftFeatureDetector detector(SIFT_DETECTOR_PARAMETER);
	SiftDescriptorExtractor extractor;

	detector.detect(_cardMatrix, _keyPoints);
	extractor.compute(_cardMatrix, _keyPoints, _descriptors);
}

void Card::setName(string newName) {
	_name = newName;
}

void Card::setSuit(string newSuit) {
	_suit = newSuit;
}

void Card::setValue(int newValue) {
	_value = newValue;
}

string Card::getName() {
	return _name;
}

string Card::getSuit() {
	return _suit;
}

int Card::getValue() {
	return _value;
}

Mat Card::getWinnerHomography() {
	return _winnerHomography;
}

Mat Card::getLoserHomography() {
	return _loserHomography;
}

Mat Card::getDescriptors() {
	return _descriptors;
}

Card Card::whoIsWinner(vector<Card> cards) {

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

Card::~Card() {
}