#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

class Card
{
public:
	Card(){}
	Card(Mat mat, Mat w, Mat l);
	Card(std::string name, Mat mat);
	~Card();

	std::string _name;
	Mat _cardMatrix;
	vector<KeyPoint> _keyPoints;
	Mat _descriptors;
	std::string _suit;
	int _value;
	Mat _winnerHomography;
	Mat _loserHomography;

	static Card whoIsWinner(vector<Card> cards);
};

Card::Card(Mat mat, Mat w, Mat l){

	_name = "Unknown";
	_cardMatrix = mat;

	_winnerHomography = w;
	_loserHomography = l;

	//Calculade this card's keypoints and descriptors
	SiftFeatureDetector detector(400);
	SiftDescriptorExtractor extractor;

	detector.detect(_cardMatrix, _keyPoints);
	extractor.compute(_cardMatrix, _keyPoints, _descriptors);
}

Card::Card(std::string name, Mat mat)
{
	_name = name;
	_cardMatrix = mat;

	size_t underscore = _name.find_first_of('_');

	_value = atoi(_name.substr(0, underscore).c_str());
	_suit = _name.substr(underscore +1);

	//Calculade this card's keypoints and descriptors
	SiftFeatureDetector detector(400);
	SiftDescriptorExtractor extractor;

	detector.detect(_cardMatrix, _keyPoints);
	extractor.compute(_cardMatrix, _keyPoints, _descriptors);
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

Card::~Card()
{
}