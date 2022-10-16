# PTX-ISA
CUDA PTX-ISA Document 中文翻译版

参考官方文档[Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)进行的翻译学习

其中PTX版本为`7.8`

记录一下学习过程，部分内容会经过提炼加上一些自己的理解。

# 第1章 Intruduction
## 1.1 Scalable Data-Parallel Computing using GPUS
PTX定义了一套抽象设备层面的ISA用于通用的并行编程指令。让开发人员可以忽略掉具体的目标设备指令集差异，进行通用的开发。

[ps: 和LLVM IR的设计定位相似]

## 1.2 Goals of PTX
 - 提供了一套覆盖多各种GPU架构的稳定ISA。
 - 可以提供近似native的性能（个人理解基本约等于汇编指令，写的逻辑基本就是机器执行逻辑）。
 - 为C/C++和其他编译器提供与目标设备架构无关的ISA。
 - 为应用和中间件开发人员提供了易用的ISA。
 - 为优化代码的生成器和转换器提供了通用ISA。
 - 简化库、性能内核和体系结构测试的手工编码。
 - 提供了可扩展的编程模型，涵盖多种架构的GPU。

## 1.3 PTX ISA Version 7.8
__`7.8`版本有如下新特性__：

1. 新增支持`sm_90`和`sm_89_`架构的支持；
2. 扩展`bar`和`barrier`指令以支持可选的范围限定符`.cta`；
3. 扩展空间限定符`.shared`支持可选的子限定符`::cta`;
4. 新增`movmatrix`指令，支持warp内寄存器进行矩阵转置；
5. 新增`stmatrix`指令，支持将一个或多个矩阵存入共享内存中；
6. 扩展`.f64`浮点类型`mma`操作，支持形状`.m16n8k4`、`.m16n8k8`和`.m16n8k16`。
7. 扩展`bf16`数据类型的`add`,`sub`, `mul`, `set`, `setp`, `cvt`, `tanh`, `ex2`, `atom`, `red`指令
8. 新增可选浮点格式`.e4m3`和`.e5m2`；（应该是用与8bit浮点）
9. 扩展`cvt`指令以支持`.e4m3`和`.e5m2`浮点格式的转换；
10. 新增`griddepcontrol`指令，作为交流空间以控制存在依赖的线程网格的执行；
11. 新增`mbarrier`指令，可允许在一个新的阶段完成`try_wait`检查操作；
12. 新增对新线程组`cluster`的支持，cluster是由多个CTA(Cooperative Thread Array)组成；
13. 为`cluster`新增`fence`,`membar`, `ld`, `st`, `atom`, `red` 指令；
14. 为`cluster`额外所需的共享空间状态添加支持；
15. 为`.shared`添加加`::cluster`子限定符，表明cluster-level可见的共享内存，并为其提供相应的`isspacep`, `cvta`, `ld`, `st`, `atom`,`red`指令；
16. 新增`mapa`指令，用于将共享内存中的地址映射到相应的地址，地址位于cluster中不同的cta中。
17. 新增`getctarank`指令，以查询包含所给地址的CTA的位置；
18. 新增`barrier.cluster`同步指令；
19. 扩展内存一致性模型以覆盖cluster域；
20. 新增cluster相关的特殊寄存器，包括：`%is_explicit_cluster`,
`%clusterid`, `%nclusterid`, `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`,
`%cluster_nctarank`;
21. 新增了cluster维度相关指令，包括：`.reqnctapercluster`, `.explicitcluster`,
`.maxclusterrank`。

# 第2章 Programming Model
## 2.1 A Highly Multithreaded Coprocessor
GPU是可以并行执行打量线程的设备，可协助CPU分担大数据量的计算工作。
## 2.2 Thread Hierarchy
执行GPU内核函数的线程被划分为**线程网格(Grid)**，而Grid又可以再向下划分为Cluster和CTA。
### 2.2.1 Cooperative Thread Arrays
在PTX的概念中，CTA是一组可以相互通信的线程所组成的线程块，对应CUDA中的Thread Block。

在CTA中同样有warp的概念，warp是CTA的最小执行线程集合，这个概念就不多赘述了。

如下图所示：
![Figure1](./images/fig1.png)
### 2.2.2 Cluster of Coopperative Thread Arrarys
Cluster是由多个CTA组成，设置Cluster大小是可选的，默认是1x1x1的大小。

其中也有特定的符号可以查询CTA的id等。存放在特殊寄存器中。

需要注意的是目前只在`sm_90`或以上的硬件架构中才支持这一概念。

如下图所示：
![Figure2](./images/fig2.png)

### 2.2.3 Grid of Cluster
Grid是最高的线程等级，包含了多个Cluster。

其中也有特定的符号可以查询Cluster的id等。存放在特殊寄存器中。
## 2.3 Memory Hierarchy
以`sm_90`架构为例，因为引入了Cluster的概念，其中的内存分类如下图所示：

![Figure3](./images/fig3.png)

主要分为以下几种：
 - global memory，可读可写，线程共享；
 - constant memory，只读，线程共享；
 - param memory
 - texture，只读，cached；
 - surface，可读可写，cached；
 - shared memory，CTA中线程共享；
 - local memory，线程独占；

# 第3章 PTX Machine Modell
## 3.1 A Set of SIMT Multiprocessors 
## 3.2 Independent Thread Scheduling
## 3.3 On-chip Shared Memory

# 第4章 Syntax
## 4.1 Source Format
## 4.2 Comments
## 4.3 Statements
### 4.3.1 Directive Statements·
### 4.3.2 Instruction Statements
## 4.4 Identifiers
## 4.5 Constants
### 4.5.1 Integer Constants
### 4.5.2 Floating-Point Constants
### 4.5.3 Predicate Constants
### 4.5.4 Constant Expressions
### 4.5.5 Integer Constant Expression Evaluation
### 4.5.6  Summary of Constant Expression Evaluation
Rules
