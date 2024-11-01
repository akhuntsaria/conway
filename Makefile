all: clean opengl-conway

opengl-conway: main.c
	gcc -o $@ $^ -lglfw -lGL -lGLEW -Wall -O3

clean:
	rm -f $(OBJ) $(TARGET)
