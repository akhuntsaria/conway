all: clean opengl-conway

opengl-conway: main.c
	gcc -o $@ $^ -lglfw -lGL -Wall -O3 

opengl-conway-debug: main.c
	gcc -o $@ $^ -lglfw -lGL -Wall -O3 -g


opengl-conway-slow: main.c
	gcc -o $@ $^ -lglfw -lGL -Wall 

clean:
	rm -f $(OBJ) $(TARGET)
