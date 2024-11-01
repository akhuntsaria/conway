#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef unsigned char u8;

int width = 3000,
    height = 2000;

void set_live(u8* cells, int x, int y) {
    cells[y * width + x] = 1;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main(void) {
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Conway", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);    
    glPointSize(1);

    u8 cells[width * height],
        buffer[width * height];

    printf("Size: %d\n", (int)sizeof(cells));

    memset(cells, 0, sizeof(cells));

    int hw = width / 2,
        hh = height / 2;
    set_live(cells, hw + 0, hh + 0);
    set_live(cells, hw + 2, hh + 0);
    set_live(cells, hw + 2, hh + 1);
    set_live(cells, hw + 4, hh + 2);
    set_live(cells, hw + 4, hh + 3);
    set_live(cells, hw + 4, hh + 4);
    set_live(cells, hw + 6, hh + 3);
    set_live(cells, hw + 6, hh + 4);
    set_live(cells, hw + 6, hh + 5);
    set_live(cells, hw + 7, hh + 4);

    //srand(time(NULL));
    int gen = 0; 
    while (1) {         
        glBegin(GL_POINTS);
        glPointSize(1.0f);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (cells[y * width + x]) {
                    glColor3f(1.0f, 1.0f, 1.0f);
                } else {
                    glColor3f(0.0f, 0.0f, 0.0f);
                }
                
                glVertex2i(x, y);
            }
        }
        glEnd();

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) break;

        glfwSwapBuffers(window);
        glfwPollEvents();
        //glfwWaitEventsTimeout(1.0f / 60.0f);
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                u8 live_ne = 0;
                for (int i = -1;i <= 1;i++) {
                    for (int j = -1;j <= 1;j++) {
                        int nx = x + i, ny = y + j;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        live_ne += cells[ny * width + nx];
                    }
                }


                u8 live = live_ne == 2 || live_ne == 3;
                buffer[y * width + x] = live;
                //pixels[y * width + x] = rand() % 2;
            }
        }
        memcpy(cells, buffer, sizeof(buffer));

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + 
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;
        printf("Gen %d took %.0fms\n", gen++, elapsed);
    }

    glfwTerminate();
    return 0;
}

