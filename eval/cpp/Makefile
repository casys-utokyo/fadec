CC := g++
CFLAGS := -Wall -std=c++11 -O3
EIGEN := -I /usr/include/eigen3

main: functional.o dataset_loader.o keyframe_buffer.o utils.o main.o
	$(CC) $(CFLAGS) functional.o dataset_loader.o keyframe_buffer.o utils.o main.o

main.o: main.cpp
	$(CC) $(CFLAGS) $(EIGEN) -c main.cpp

utils.o: utils.cpp
	$(CC) $(CFLAGS) $(EIGEN) -c utils.cpp

keyframe_buffer.o: keyframe_buffer.cpp
	$(CC) $(CFLAGS) -c keyframe_buffer.cpp

dataset_loader.o: dataset_loader.cpp
	$(CC) $(CFLAGS) -c dataset_loader.cpp

functional.o: functional.cpp
	$(CC) $(CFLAGS) -c functional.cpp

clean:
	$(RM) *.o *.out