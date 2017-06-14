all: CImg.h a3.cpp Classifier.h NearestNeighbor.h SVM.h Haas.h Eigen.h CNN.h Bags.h
	g++ a3.cpp -o a3 -lX11 -lpthread -I. -Isiftpp -O3 siftpp/sift.cpp -std=c++11

clean:
	rm a3
