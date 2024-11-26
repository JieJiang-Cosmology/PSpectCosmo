# Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
# This code is licensed under the MIT License.
# See the LICENSE file in the project root for license information.

# Makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lm -lfftw3 -lfftw3_threads -lhdf5_cpp -lhdf5

# Object files
OBJS = pspectcosmo.o initialize.o evolution.o output.o

# Target
TARGET = pspectcosmo

DIR = output
SNAPS_DIR = Snapshots


# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

cleardata:
	rm -f $(OBJS) $(TARGET) $(DIR)/*.txt $(DIR)/$(SNAPS_DIR)/*.h5

new: clean all

newrun: cleardata all 
	./$(TARGET)

.PHONY: all clean cleardata new
