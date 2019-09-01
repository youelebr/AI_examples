CXX=g++
CXXFLAGS=-std=c++11 `pkg-config opencv --cflags --libs`

all: face_detection

face_detection:	
	$(CXX) -o $@ $@.cpp $(CXXFLAGS)

clean:
	rm -rf face_detection
