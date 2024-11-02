#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstddef>
#include <cstdlib>
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

const int GUNS = 1000,
    GLIDERS = 10000,
    PAN_SPEED = 10;

int buff_w = 2000,
    buff_h = 1000,
    window_w = 1500,
    window_h = 800,
    pan_x = 0,
    pan_y = 0;

u8 pause = 0,
    quit = 0;

size_t buff_size = (size_t)buff_w * buff_h,
    window_size = (size_t)window_w * window_h;

dim3 block_size = dim3(32, 32, 1);
dim3 grid_size = dim3((buff_w + block_size.x - 1) / block_size.x, (buff_h + block_size.y - 1) / block_size.y, 1);

// Copy only what's shown, device to host
void partial_cpy_dth(u8* host_buff, u8* dev_buff) {
    size_t start_x = (buff_w - window_w) / 2 + pan_x,
        start_y = (buff_h - window_h) / 2 + pan_y;

    for (int y = 0; y < window_h; ++y) {
        size_t src_i = (start_y + y) * buff_w + start_x,
            dst_i = y * window_w;
        cudaMemcpy(&host_buff[dst_i], &dev_buff[src_i], window_w, cudaMemcpyDeviceToHost);
    }
}

void partial_cpy_htd(u8* dev_buff, u8* host_buff) {
    size_t start_x = (buff_w - window_w) / 2 + pan_x,
        start_y = (buff_h - window_h) / 2 + pan_y;

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
    // Column-major order
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

int rando(int min, int max) {
    return (rand() % (max - min + 1)) + min;
}

void initial_state(u8* dev_buff) {
    srand(time(NULL));

    u8 gun[64] = {
        255,0,  255,0,  0,  0,  0,  0,
        0,  0,  255,0,  0,  0,  0,  0,
        0,  0,  0,  0,  255,0,  0,  0,
        0,  0,  0,  0,  255,0,  255,0,
        0,  0,  0,  0,  255,0,  255,255,
        0,  0,  0,  0,  0,  0,  255,0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
    };

    int border_h = buff_h / 50,
        border_w = buff_w / 50;

    for (int figure = 0;figure < GUNS;figure++) {
        int start_x = rando(border_w, buff_w - border_w),
            start_y = rando(border_h, buff_h - border_h);

        for (int i = 0; i < 8; i++) {
            size_t src_i = i * 8,
                dst_i = (start_y + i) * buff_w + start_x;
            cudaMemcpy(&dev_buff[dst_i], &gun[src_i], 8, cudaMemcpyHostToDevice);
        }
    }

    u8 glider1[9] = {
        0,  255,255,
        255,0,  255,
        0,  0,  255,
    }, glider2[9] = {
        0,  0,  255,
        255,0,  255,
        0,  255,255,
    };

    for (int figure = 0;figure < GLIDERS;figure++) {
        int start_x = rando(border_w, buff_w - border_w),
            start_y = rando(border_h, buff_h - border_h);

        u8* glider = rand() % 2 ? glider1 : glider2;

        for (int i = 0; i < 3; i++) {
            size_t src_i = i * 3,
                dst_i = (start_y + i) * buff_w + start_x;
            cudaMemcpy(&dev_buff[dst_i], &glider[src_i], 3, cudaMemcpyHostToDevice);
        }
    }
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_LEFT) {
            pan_x = pan_x - PAN_SPEED;
        } else if (key == GLFW_KEY_RIGHT) {
            pan_x = pan_x + PAN_SPEED;
        } else if (key == GLFW_KEY_UP) {
            pan_y = pan_y + PAN_SPEED;
        } else if (key == GLFW_KEY_DOWN) {
            pan_y = pan_y - PAN_SPEED;
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

    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(window_w, window_h, "Conway", NULL, NULL);
    if (!window) return 1;

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    glfwSetKeyCallback(window, key_callback);

    u8 *host_buff, *dev_buff, *dev_buff_copy;    
    cudaHostAlloc((void**)&host_buff, window_size, cudaHostAllocDefault);
    cudaMalloc(&dev_buff, buff_size);
    cudaMalloc(&dev_buff_copy, buff_size);

    cudaMemset(dev_buff, 0, buff_size);
    initial_state(dev_buff);

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
        printf("  pan x %d\n", pan_x); 
        printf("  pan y %d\n", pan_y); 

        glEnable(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, window_w, window_h, 0, GL_RED, GL_UNSIGNED_BYTE, host_buff);

        glBegin(GL_QUADS);
            glTexCoord2f(.0f, .0f); glVertex2f(-1.f, -1.f);
            glTexCoord2f(1.f, .0f); glVertex2f(1.f, -1.f);
            glTexCoord2f(1.f, 1.f); glVertex2f(1.f, 1.f);
            glTexCoord2f(.0f, 1.f); glVertex2f(-1.f, 1.f);
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

