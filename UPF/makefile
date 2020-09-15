CFLAGS = -Wall -std=c++14 `pkg-config --cflags --libs opencv` -c -O3 -g -o
LFLAGS = -Wall -std=c++14 `pkg-config --cflags --libs opencv` -g -O3 -o
PFLOC = ../PFV4/

track: tracker.o acquireData.o sequence.o PF.o UPF.o detectSquare.o
	g++ $^ $(LFLAGS) $@

UPF.o: UPF.cpp UPF.hpp
	g++ $< $(CFLAGS) $@

PF.o: $(PFLOC)$ PF.cpp $(PFLOC)$ PF.hpp
	g++ $< $(CFLAGS) $@

sequence.o: $(PFLOC)$ sequence.cpp $(PFLOC)$ sequence.hpp
	g++ $< $(CFLAGS) $@

acquireData.o: $(PFLOC)$ acquireData.cpp $(PFLOC)$ acquireData.hpp
	g++ $< $(CFLAGS) $@

detectSquare.o: $(PFLOC)$ detectSquare.cpp $(PFLOC)$ detectSquare.hpp
	g++ $< $(CFLAGS)$ $@

run:
	./track

clean:
	rm *.o track

