#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstddef>
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

int buff_w = 60000,
    buff_h = 60000,
    window_w = 1500,
    window_h = 800,
    off_x = 0,
    off_y = 0,
    off_speed = 10;

//TODO fix
float zoom = 1.0f;

u8 pause = 0,
    quit = 0;

size_t buff_size = (size_t)buff_w * buff_h,
    window_size = (size_t)window_w * window_h;

dim3 block_size = dim3(32, 32, 1);
dim3 grid_size = dim3((buff_w + block_size.x - 1) / block_size.x, (buff_h + block_size.y - 1) / block_size.y, 1);

// Copy only what's shown, device to host
void partial_cpy_dth(u8* host_buff, u8* dev_buff) {
    size_t start_x = (buff_w - window_w) / 2 + off_x,
        start_y = (buff_h - window_h) / 2 + off_y;

    for (int y = 0; y < window_h; ++y) {
        size_t src_i = (start_y + y) * buff_w + start_x,
            dst_i = y * window_w;
        cudaMemcpy(&host_buff[dst_i], &dev_buff[src_i], window_w, cudaMemcpyDeviceToHost);
    }
}

void partial_cpy_htd(u8* dev_buff, u8* host_buff) {
    size_t start_x = (buff_w - window_w) / 2 + off_x,
        start_y = (buff_h - window_h) / 2 + off_y;

    for (int y = 0; y < window_h; ++y) {
        size_t src_i = y * window_w,
            dst_i = (start_y + y) * buff_w + start_x;
        cudaMemcpy(&dev_buff[dst_i], &host_buff[src_i], window_w, cudaMemcpyHostToDevice);
    }
}

__global__ void update_cells_kernel(u8* buff, u8* buff_copy, int width, int height) {
    long x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    u8 live_ne = 0;
    for (int i = -1;i <= 1;i++) {
        for (int j = -1;j <= 1;j++) {
            long nx = x + i, ny = y + j;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height || (nx == x && ny == y)) continue;
            live_ne += buff_copy[ny * width + nx] == 255;
        }
    }

    u8 col;
    if (buff_copy[y * width + x] == 255) {
        col = live_ne == 2 || live_ne == 3 ? 255 : 0;;
    } else {
        col = live_ne == 3 ? 255 : 0;
    }
    buff[y * width + x] = col;
}

void dev_update_cells(u8* host_buff, u8* dev_buff, u8* dev_buff_copy) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dev_buff_copy, dev_buff, buff_size, cudaMemcpyDeviceToDevice);
    update_cells_kernel <<<grid_size, block_size>>>(dev_buff, dev_buff_copy, buff_w, buff_h);
    cudaCheck();
    partial_cpy_dth(host_buff, dev_buff);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  Dev ops took %.0fms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void set_live(u8* buff, int x, int y) {
    if (x < window_w && y < window_h) {
        buff[y * window_w + x] = 255;
    }
}

//TODO entire width of the dev_buff
void initial_state(u8* host_buff) {
    float hh = window_h / 2.0f,
        hw = window_w / 2.0f;
    // Guns
    for (int offset_x = 50;offset_x < window_w;offset_x += 100) {
        set_live(host_buff, offset_x + 0, hh + 0);
        set_live(host_buff, offset_x + 2, hh + 0);
        set_live(host_buff, offset_x + 2, hh + 1);
        set_live(host_buff, offset_x + 4, hh + 2);
        set_live(host_buff, offset_x + 4, hh + 3);
        set_live(host_buff, offset_x + 4, hh + 4);
        set_live(host_buff, offset_x + 6, hh + 3);
        set_live(host_buff, offset_x + 6, hh + 4);
        set_live(host_buff, offset_x + 6, hh + 5);
        set_live(host_buff, offset_x + 7, hh + 4);
    }

    for (int offset_y = 50;offset_y < window_h;offset_y += 100) {
        set_live(host_buff, hw + 0, offset_y + 0);
        set_live(host_buff, hw + 2, offset_y + 0);
        set_live(host_buff, hw + 2, offset_y + 1);
        set_live(host_buff, hw + 4, offset_y + 2);
        set_live(host_buff, hw + 4, offset_y + 3);
        set_live(host_buff, hw + 4, offset_y + 4);
        set_live(host_buff, hw + 6, offset_y + 3);
        set_live(host_buff, hw + 6, offset_y + 4);
        set_live(host_buff, hw + 6, offset_y + 5);
        set_live(host_buff, hw + 7, offset_y + 4);
    }
    
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        /*if (key == GLFW_KEY_EQUAL) {
            zoom += .3f;
        } else if (key == GLFW_KEY_MINUS) {
            zoom -= .3f;
        } else*/ if (key == GLFW_KEY_LEFT) {
            off_x -= off_speed;
        } else if (key == GLFW_KEY_RIGHT) {
            off_x += off_speed;
        } else if (key == GLFW_KEY_UP) {
            off_y += off_speed;
        } else if (key == GLFW_KEY_DOWN) {
            off_y -= off_speed;
        } else if (key == GLFW_KEY_Q) {
            quit = 1;
        } else if (key == GLFW_KEY_P) {
            pause = !pause;
        }
    }
}

int main(void) {
    printf("Buffer size: %zd\n", buff_size);
    printf("Window data size: %zd\n", window_size);
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

    u8 *host_buff, *dev_buff, *dev_buff_copy;
    
    cudaHostAlloc((void**)&host_buff, window_size, cudaHostAllocDefault);
    cudaCheck();
    memset(host_buff, 0, window_size);
    initial_state(host_buff);

    cudaMalloc(&dev_buff, buff_size);
    cudaMemset(dev_buff, 0, buff_size);
    cudaMalloc(&dev_buff_copy, buff_size);
    partial_cpy_htd(dev_buff, host_buff);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int gen = 0; 
    while (!glfwWindowShouldClose(window) && !quit) {
        glfwPollEvents();

        if (pause) continue;

        printf("Gen %d\n", ++gen);
        printf("  x offset %d\n", off_x); 
        printf("  y offset %d\n", off_y); 

        glEnable(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, window_w, window_h, 0, GL_RED, GL_UNSIGNED_BYTE, host_buff);
        
        float halfZoom = 0.5f / zoom,
            left = 0.5f - halfZoom,
            right = 0.5f + halfZoom,
            bottom = 0.5f - halfZoom,
            top = 0.5f + halfZoom;

        glBegin(GL_QUADS);
            glTexCoord2f(left, bottom); glVertex2f(-1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, bottom); glVertex2f(1.0f * zoom, -1.0f * zoom);
            glTexCoord2f(right, top); glVertex2f(1.0f * zoom, 1.0f * zoom);
            glTexCoord2f(left, top); glVertex2f(-1.0f * zoom, 1.0f * zoom);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glfwSwapBuffers(window);

        dev_update_cells(host_buff, dev_buff, dev_buff_copy);
    }

    cudaFree(dev_buff);
    cudaFree(dev_buff_copy);

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

