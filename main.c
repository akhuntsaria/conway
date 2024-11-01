#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef unsigned char u8;

int buff_w = 10000,
    buff_h = 10000,
    window_w = 800,
    window_h = 800;

float zoom = 1.0f,
    off_x = .0f,
    off_y = .0f,
    off_speed = .005f;

u8 pause = 0,
    quit = 0;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void set_live(u8* cells, int x, int y) {
    cells[(y * buff_w + x) * 3] = 255;
    cells[(y * buff_w + x) * 3 + 1] = 255;
    cells[(y * buff_w + x) * 3 + 2] = 255;
}

void initial_state(u8* cells) {
    int hw = buff_w / 2,
        hh = buff_h / 2;
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
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_EQUAL) {
            zoom += .3f;
            printf("New zoom: %.2f\n", zoom);
        } else if (key == GLFW_KEY_MINUS) {
            zoom -= .3f;
            printf("New zoom: %.2f\n", zoom);
        } else if (key == GLFW_KEY_LEFT) {
            off_x -= off_speed / zoom;
            printf("New x offset: %.3f\n", off_x);
        } else if (key == GLFW_KEY_RIGHT) {
            off_x += off_speed / zoom;
            printf("New x offset: %.3f\n", off_x);
        } else if (key == GLFW_KEY_UP) {
            off_y += off_speed / zoom;
            printf("New y offset: %.3f\n", off_y);
        } else if (key == GLFW_KEY_DOWN) {
            off_y -= off_speed / zoom;
            printf("New y offset: %.3f\n", off_y);
        } else if (key == GLFW_KEY_Q) {
            quit = 1;
        } else if (key == GLFW_KEY_P) {
            pause = !pause;
        }
    }
}

void update_cells(u8* buff, u8* cells, int gen) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int x = 0; x < buff_w; x++) {
        for (int y = 0; y < buff_h; y++) {
            u8 live_ne = 0;
            for (int i = -1;i <= 1;i++) {
                for (int j = -1;j <= 1;j++) {
                    int nx = x + i, ny = y + j;
                    if (nx < 0 || nx >= buff_w || ny < 0 || ny >= buff_h) continue;
                    live_ne += cells[(ny * buff_w + nx) * 3] == 255;
                }
            }


            u8 col = live_ne == 2 || live_ne == 3 ? 255 : 0;
            buff[(y * buff_w + x) * 3] = col;
            buff[(y * buff_w + x) * 3 + 1] = col;
            buff[(y * buff_w + x) * 3 + 2] = col;
        }
    }

    memcpy(cells, buff, buff_w * buff_h * 3);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + 
                    (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Gen %d, %.0fms\n", gen, elapsed);
}

int main(void) {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(window_w, window_h, "Conway", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    glfwSetKeyCallback(window, key_callback);

    u8 *cells = malloc(buff_w * buff_h * 3),
        *buff = malloc(buff_w * buff_h * 3);

    memset(cells, 0, buff_w * buff_h * 3);

    initial_state(cells);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buff_w, buff_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cells);

    int gen = 0; 
    while (!glfwWindowShouldClose(window) && !quit) {  
        glfwWaitEventsTimeout(1.0f / 60.0f);

        if (pause) {
            continue;
        }

        glClear(GL_COLOR_BUFFER_BIT);
    
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buff_w, buff_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cells);
        
        float halfZoom = 0.5f / zoom;
        float left = off_x + 0.5f - halfZoom;
        float right = off_x + 0.5f + halfZoom;
        float bottom = off_y + 0.5f - halfZoom;
        float top = off_y + 0.5f + halfZoom;

        glBegin(GL_QUADS);
            glTexCoord2f(left, bottom); glVertex2f(-1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, bottom); glVertex2f(1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, top); glVertex2f(1.0f * zoom, 1.0f * zoom);
            glTexCoord2f(left, top); glVertex2f(-1.0f * zoom, 1.0f * zoom);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        
        glfwSwapBuffers(window);
        
        update_cells(buff, cells, gen++);
    }

    free(buff);
    free(cells);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

