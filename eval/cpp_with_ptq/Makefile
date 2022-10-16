CC := g++
CFLAGS := -Wall -std=c++11 -O3
EIGEN := -I /usr/include/eigen3

main: functional.o dataset_loader.o keyframe_buffer.o utils.o run-testing-online.o
	$(CC) $(CFLAGS) functional.o dataset_loader.o keyframe_buffer.o utils.o run-testing-online.o

run-testing-online.o: run-testing-online.cpp
	$(CC) $(CFLAGS) $(EIGEN) -c run-testing-online.cpp

utils.o: utils.cpp
	$(CC) $(CFLAGS) $(EIGEN) -c utils.cpp

keyframe_buffer.o: keyframe_buffer.cpp
	$(CC) $(CFLAGS) -c keyframe_buffer.cpp

dataset_loader.o: dataset_loader.cpp
	$(CC) $(CFLAGS) -c dataset_loader.cpp

functional.o: functional.cpp
	$(CC) $(CFLAGS) -c functional.cpp

conv-test:
	$(CC) $(CFLAGS) conv-test.cpp && ./a.out

mnasnet-test:
	$(CC) $(CFLAGS) mnasnet-test.cpp && ./a.out

model-test:
	$(CC) $(CFLAGS) model-test.cpp && ./a.out

clean:
	$(RM) *.o *.out