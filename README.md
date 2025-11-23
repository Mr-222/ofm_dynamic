# ONE-STEP FLOW MAPS FOR REAL-TIME FLUID SIMULATION WITH DYNAMIC BOUNDARIES
This repository contains the code for my [master's thesis](Yutong_Sun_MSCS_Thesis.pdf) at Georgia Tech.

## Build Instructions
We use xmake for cross-platform compilation. We successfully compiled the code on machines with Windows 11 and Nvidia RTX 4080. 

### 1. Clone the repository
```
git clone https://github.com/Mr-222/ofm_dynamic.git
```
### 2. Update submodule (for Vulkan Renderer and AMPCG Poisson Solver)
```
git submodule update --init --recursive
```
### 3. Dependencies
* xmake
* C++ 20
* Cuda 12.6
* Vulkan
* sed, gnuutils

### 4. Build

At proj/

```
xmake build
```

## Run

At proj/sim_render, run the executable file in proj/dynamic_obstacle:
```
.\build\dynamic_obstacle.exe
```

You are expected to see:
![](image/rotating_octa.png)


## Configuration

In proj/dynamic_obstacle/config