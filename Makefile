all: clean opengl-conway

opengl-conway: main.cu
	nvcc -arch=sm_89 -O3 -Xcompiler -Wall -Iinclude -Llib -lGLEW -lglfw -lGL -o $@ $^

clean:
	rm -f $(OBJ) $(TARGET)
