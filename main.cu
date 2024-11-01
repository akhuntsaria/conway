#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define cudaCheck() { _cudaCheck(__FILE__, __LINE__); }
inline void _cudaCheck(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(error);
    }
}

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

size_t buff_size = (size_t)buff_w * buff_h * 3;

dim3 block_size = dim3(32, 32, 1);
dim3 grid_size = dim3((buff_w + block_size.x - 1) / block_size.x, (buff_h + block_size.y - 1) / block_size.y, 1);
size_t shared_size = block_size.x * block_size.y * 3;

__global__ void update_cells_kernel(u8* buff, u8* buff_copy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    u8 live_ne = 0;
    for (int i = -1;i <= 1;i++) {
        for (int j = -1;j <= 1;j++) {
            int nx = x + i, ny = y + j;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height || (nx == x && ny == y)) continue;
            live_ne += buff_copy[(ny * width + nx) * 3] == 255;
        }
    }

    u8 col;
    if (buff_copy[(y * width + x) * 3] == 255) {
        col = live_ne == 2 || live_ne == 3 ? 255 : 0;;
    } else {
        col = live_ne == 3 ? 255 : 0;
    }
    buff[(y * width + x) * 3] = col;
    buff[(y * width + x) * 3 + 1] = col;
    buff[(y * width + x) * 3 + 2] = col;
}

void dev_update_cells(u8* buff, u8* dev_buff, u8* dev_buff_copy, int gen) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dev_buff_copy, buff, buff_size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Gen %d, %.0fms\n", gen, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    update_cells_kernel <<<grid_size, block_size, shared_size>>>(dev_buff, dev_buff_copy, buff_w, buff_h);
    cudaCheck();
    cudaMemcpy(buff, dev_buff, buff_size, cudaMemcpyDeviceToHost);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void set_live(u8* buff, int x, int y) {
    buff[(y * buff_w + x) * 3] = 255;
    buff[(y * buff_w + x) * 3 + 1] = 255;
    buff[(y * buff_w + x) * 3 + 2] = 255;
}

void initial_state(u8* buff) {
    int hw = buff_w / 2,
        hh = buff_h / 2;
    set_live(buff, hw + 0, hh + 0);
    set_live(buff, hw + 2, hh + 0);
    set_live(buff, hw + 2, hh + 1);
    set_live(buff, hw + 4, hh + 2);
    set_live(buff, hw + 4, hh + 3);
    set_live(buff, hw + 4, hh + 4);
    set_live(buff, hw + 6, hh + 3);
    set_live(buff, hw + 6, hh + 4);
    set_live(buff, hw + 6, hh + 5);
    set_live(buff, hw + 7, hh + 4);
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

int main(void) {
    printf("Buffer size: %zd\n", buff_size);
    printf("Block Size: (%d, %d, %d)\n", block_size.x, block_size.y, block_size.z);
    printf("Grid Size: (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z);

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
    if (glewInit() != GLEW_OK) return -1;

    glfwSetKeyCallback(window, key_callback);

    u8 *buff;
    cudaHostAlloc((void**)&buff, buff_size, cudaHostAllocDefault); // Pinned memory
    cudaCheck();

    memset(buff, 0, buff_size);
    initial_state(buff);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buff_w, buff_h, 0, GL_RGB, GL_UNSIGNED_BYTE, buff);

    u8 *dev_buff, *dev_buff_copy;
    cudaMalloc(&dev_buff, buff_size);
    cudaMalloc(&dev_buff_copy, buff_size);

    int gen = 0; 
    while (!glfwWindowShouldClose(window) && !quit) {  
        glfwPollEvents();

        if (pause) continue;

        glClear(GL_COLOR_BUFFER_BIT);
    
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buff_w, buff_h, 0, GL_RGB, GL_UNSIGNED_BYTE, buff);
        
        float halfZoom = 0.5f / zoom,
            left = off_x + 0.5f - halfZoom,
            right = off_x + 0.5f + halfZoom,
            bottom = off_y + 0.5f - halfZoom,
            top = off_y + 0.5f + halfZoom;

        glBegin(GL_QUADS);
            glTexCoord2f(left, bottom); glVertex2f(-1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, bottom); glVertex2f(1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, top); glVertex2f(1.0f * zoom, 1.0f * zoom);
            glTexCoord2f(left, top); glVertex2f(-1.0f * zoom, 1.0f * zoom);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glfwSwapBuffers(window);

        dev_update_cells(buff, dev_buff, dev_buff_copy, gen++);
    }

    cudaFreeHost(buff);
    cudaFree(dev_buff);
    cudaFree(dev_buff_copy);

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

