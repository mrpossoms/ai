SRCS=driver.cpp net.cpp rl.cpp
OBJS=$(patsubst %.cpp,%.o,$(SRCS))

.PHONY: list-objs
list-objs:
	echo $(OBJS)

%.o: %.cpp
	g++ -g -std=c++17 -I gitman_sources -c $< -o $@

driver: $(OBJS)
	g++ -g -std=c++17 -I gitman_sources $^ -o $@ -larmadillo -lncurses -lpthread -lm

.PHONY: clean
clean:
	rm -f *.o driver
