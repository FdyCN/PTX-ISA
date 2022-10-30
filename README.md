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
 - constant memory，只读，cached，线程共享；
 - texture，只读，cached；
 - surface，可读可写，cached；
 - shared memory，CTA中线程共享；
 - local memory，线程独占；

# 第3章 PTX Machine Model
## 3.1 A Set of SIMT Multiprocessors 
GPU硬件模型如下图所示：
![Figure4](./images/fig4.png)

## 3.2 Independent Thread Scheduling
在`Volta`架构之前，一个warp内的32个线程因为共用一个程序计数器，通过active mask来区别active thread。

但是从`Volta`架构开始，支持了warp内的线程独立调度，每个线程都有自己独立的程序计数器，也就是当出现一个warp内的线程分化的时候，允许不同的线程做不同的事情，不再阻塞。

开发者在编写`Volta`及以上架构的PTX代码时，需要特别留意因为独立线程调度操作引起的**向下兼容性问题**。

## 3.3 On-chip Shared Memory
根据Figure4中的信息显示，每个Multiprocessor可以利用的片上内存主要分为以下四种：

1. 每个processor都有一组32-bit的本地寄存器；
2. 每个processor共享的`shared memory`，其拥有并行数据缓存；
3. 每个processor可通过共享的只读cache，加速读取设备的指定常量存储区域`constant memory`，内存有限；
4. 每个processor可通过共享的只读cache，加速读取设备指定的存储区域`texutre`，支持多种寻址模式和数据滤波器；

需要注意的是，`local memory`和`global memory`没有专用cache加速。 

# 第4章 Syntax
PTX源程序模块带有汇编语法风格的指令操作符和操作数。通过ptxas后端编译优化器对PTX源模块进行优化、编译并生成对应的二进制对象文件。

## 4.1 Source Format
源模块是以ASCII文本形式，以`\n`进行换行。

所有空格将被忽略，除非在语言中被用于分格标记。

接受C风格的预处理标记，通过`#`标记，如：`#include, #define, #if, #ifdef, #else, #endif, #line, #file`

PTX区分大小写，关键字用小写。

每个PTX模块必须以指定PTX语言版本的`.version`指令开始，
接着是一个`.target`指令，指定假设的目标体系结构。

## 4.2 Comments
PTX的注释服从C\C++风格，使用/* 注释内容 */或`//`均可。

## 4.3 Statements
PTX中的陈述语句既包含预处理(directive)也包含指令(instruction)，以可选的指令标记开头并以分号结尾。

例子：
```
        .reg .b32 r1, r2;
        .global .f32 array[N];

start:  mov.b32 r1, %tid.x;
        shl.b32 r1, r1, 2; // shift thread id by 2 bits
        ld.global.b32 r2, array[r1]; // thread[tid] gets array[tid]
        add.f32 r2, r2, 0.5; // add 1/2
```
【PS】文档原话：“A PTX statement is either a directive or an instruction”。这里directive和instruction的区别理解了半天，最后可以理解为，directive近似预处理的东西或者说特殊字符处理，起到编译器指示作用。而instruction是指令，可以理解为发生在机器上的“动词”。

### 4.3.1 Directive Statements
PTX中支持的编译器指示如下表所示：

![Table1](./images/table1.png)

【PS】可以看到如`.reg`、`pragma`等常用的编译指示关键字。

### 4.3.2 Instruction Statements
指令由一个指令操作码和由逗号分隔的零个或多个操作数组成，并以分号结束。操作数可以是寄存器变量，常量表达式、地址表达式或指令标签名称。

指令有一个可选的判断条件作控制流的跳转。判断条件在可选的指令标记后面，在操作码前面，并被写成`@p`，其中`p`是一个条件寄存器。判断条件可以取非，写成`@!p`。

指令标记之后的字段，首先是目标操作数，后续是源操作数。

指令关键字如下表所示：

![Table2](./images/table2.png)

## 4.4 Identifiers
用户定义的标识符，服从C++的规则，字母或者下划线开头，或者以`$`开头。

PTX没有指定标识符的最大长度，并表示所有实现至少支持1024个字符。

PTX支持以`%`为前缀的变量，用于避免命名冲突，如：用户定义的变量和编译器生成的变量名。

PTX以`%`为前缀预定义了一个常量和一小部分特殊寄存器，如下表所示：

![Table3](./images/table3.png)

其中`WARP_SZ`表明了目标设备的warp大小，默认值都是32。

## 4.5 Constants
PTX支持整型和浮点常量和常量表达式。这些常数可用于数据初始化和作为指令的操作数。对于整型、浮点和位大小类型检查规则是相同的。

对于判断类型的数据和指令，允许使用整型常量，即`0`为`False`和`!0`为`True`。

### 4.5.1 Integer Constants
整型常量的大小为64位，有符号或无符号，即每个整数常量的类型为`.s64`或`.u64`。

而在指令或数据初始化中使用时，每个整整型常量会根据使用时的数据或指令类型转换为适当的大小。


整型常量可以写作十六进制、十进制、八进制、二进制，写法同C语言一直，最后加`U`表示unsigned:

```
十六进制：    0[xX]{hexdigit}+(U)
十进制：      {nonzero-digit}{digit}+(U)
八进制：      0{octal digit}+(U)
二进制：      0[bB]{bit}+(U)
```

### 4.5.2 Floating-Point Constants
浮点常量表示为64位双精度值，所有浮点常量表达式都使用64位双精度算术求值。

需要注意的是如果用十六进制表示，是表示32位单精度浮点。并且可能不会被用在常量表达式中。

浮点数的值的第一种表示，可以用一个可选的小数点和带符号的指数进行表达(应该是指：1.34e-2)。但和C\C++不同的是，在PTX里面不能通过后缀来区分浮点数的的类型，比如：1.0f。

浮点数的值的第二种表示，可以使用十六进制进行表示，如下：

```
0[fF]{hexdigit}{8} // single-precision floating point
0[dD]{hexdigit}{16} // double-precision floating point
```

举个例子：
```
mov.f32 $f3, 0F3f800000; // 1.0, 表示：$f3 = 1.0;
```

### 4.5.3 Predicate Constants
整型常量也可以作为判断数据，`0`表示`False`，`!0`表示`True`。

### 4.5.4 Constant Expressions
在PTX中，常量表达式是使用C中的操作符形成的，并使用与C中类似的规则求值，但通过限制类型和大小、删除大多数强制转换和定义完整语义来简化，以消除C中表达式求值依赖于实现的情况。(减去编译器推导数据类型等的负担)

常量表达不支持从整型到浮点数的类型转换。

常量表达式中的优先级顺序**从上到下**如小表所示，第一行执行优先级最高，同一行的优先级相同，对于多个一元操作求值的话是**从右向左**的顺序，而二元操作是**从左向右**：

![Table4](./images/table4.png)

### 4.5.5 Integer Constant Expression Evaluation
整型常量表达式，在编译时有一套规则进行推导。这些规则基于C中的规则，但它们已被简化为只适用于64位整数，并且在所有情况下都**完全定义了行为**。（可以理解为不会有二义性的表达式）

- 默认整型常数是`signed`除非需要转换为`unsigned`防止溢出，或者手动添加`U`后缀如：
```
42, 0x1234, 0123 are signed.
0xfabc123400000000, 42U, 0x1234U are unsigned
```  
- 一元加减符保留输入操作数的类型，如： 
```
+123, -1, -(-42) are signed.
-1U, -0xfabc123400000000 are unsigned.
```
- 一元操作中的取非`!`操作会产生带符号的`0`或`1`。
- 位操作中的取反操作`~`默认将源操作数是为`unsigned`，结果也为`unsigned`。
- 一些二元操作需要规范化源操作数，如果其中有一个是`unsigned`，那么需要将两个源操作数都转换为`unsigned`进行计算。**这种被称为常用算数转换**。
- 加减乘除执行计算之后，结果与源操作数的数据类型保持一致，即，有一个为`unsigned`则结果也为`unsigned`，反之则为`signed`。
- 取余`%`的操作会将操作数解释为`unsigned`，与C不同，C允许负除数。但属于**实现定义行为**
- 移位操作的第二个源操作数解释为`unsigned`，结果数据类型与第一个源操作数一致。如果是`signed`右移则为算术右移，`unsigned`为逻辑右移。
- 位与`&`，位或`|`，位异或`^`操作也服从常用数据转换规则。
- 与`&&`，或`||`，等于`==`，不等`!=`操作产生`signed`结果，值为`0`或`1`。
- 大小比较运算符(`<`、`>`、`<=`、`>=`)对于源操作数符服从常用转换规则，产生`signed`结果，值为`0`或`1`。
- 可使用`(.s64)`或`(.u64)`将表达式转换为`signed`或`unsigned`。
- 对于三元判断符`?:`，第一个源操作数必须是整型，但第二个和第三个可以是整型或者浮点型，其结果类型与选择的操作数类型一致。


### 4.5.6  Summary of Constant Expression Evaluation
下表总结了常量表达式的推导规则：

![Table5](./images/table5.png)

# 第5章 State Spaces, Types, and Variables
虽然特殊的资源在不同架构的GPU上可能是不同的，但资源种类是通用的，这些资源通过状态空间和数据类型在PTX中被抽象出来。

## 5.1 State Spaces
状态空间是具有特定特征的存储区域。所有变量都驻留在某个状态空间中。状态空间的特征包括其大小、可寻址性、访问速度、访问权限和线程之间的共享级别。

不同的状态空间如下表所示：

![Table6](./images/table6.png)

不同状态空间的性质如下表所示：

![Table7](./images/table7.png)

### 5.1.1 Register State Space
`.reg`寄存器读写速度很快，但是数量有限制，并且不同架构的寄存器数量不一样。当寄存器使用超标时，会溢出到内存中，影响读写速度。

寄存器可以是有类型的，也可以是无类型的，但是寄存器大小是被严格限制的，除了1-bit的判断符(bool)寄存器以外，还有宽度为8-bit\16-bit\32-bit\64-bit的标量寄存器，以及16-bit\32-bit\64-bit\128-bit的矢量寄存器。

8-bit寄存器最常见用途是和`ld`、`st`和`cvt`指令一起使用，或作为向量组的元素。

寄存器与其他状态空间的区别在于，它们不是完全可寻址的，也就是说，不可能引用寄存器的地址。(可以理解为仅在作用域内有效，即寄存器是栈上存储)

寄存器对于多字的读写可能会需要做边界对齐。

### 5.1.2 Special Register State Space
`.sreg`特殊寄存器是预定义的平台特殊寄存器，如grid、cluster等相关参数，所有的特殊寄存器都是预定义的。

### 5.1.3 Constant State Space
`.const`常量状态空间是由host端初始化的只读内存，通常使用`ld.const`进行访问，目前常量内存的限制为64KB。

另外还有一个640KB的常量内存，被划分为10个64KB的区域，驱动程序可以在这些区域上进行初始化数据分配，并通过指针的形式作为kernel参数传入。

但是，因为这十个常量内存区域**并不连续**，所以驱动程序在分配的时候应该保证每一块常量内存不得超过64KB，不得越界。

静态大小的常量变量有一个可选的变量初始化器。默认情况下，没有显式初始化式的常数变量被初始化为零。驱动程序分配的常量缓冲区由host初始化，并将指向这块常量内存的指针作为kernel参数传入。

#### 5.1.3.1.  Banked Constant State Space (deprecated)
被弃用的就不赘述了。

### 5.1.4.  Global State Space
`.global`全局状态空间是能够被kernel中所有线程都访问到的内存空间，使用`ld.global`、`st.global`和`atom.global`指令访问全局内存。

没有显示初始化的全局变量默认初始化为`0`。

### 5.1.5.  Local State Space
`.local`本地状态空间是每个线程私有的内存空间。通常是带缓存的标准内存。其有大小限制，因为必须按每一个线程进行分配。

使用`ld.local`、`st.local`进行本地变量的访问。

在编译的ABI的时候，我们必须将`.local`声明在函数作用域内，并且内存申请在栈上。

在不支持堆栈的实现中，所有本地内存变量都存储在固定地址中，不支持递归函数调用，并且`.local`变量可能在模块(module)作用域声明。

在PTX 3.0及一下，module-scope `.local`将默认被禁用。

### 5.1.6.  Parameter State Space 
`.param`参数状态空间主要用于以下情况：

1. 作为从host传入kernel的输入参数；
2. 在kernel执行过程中，为调用的device函数声明正式的输入和返回参数；
3. 通常可用于声明局部作用域的字节矩阵，主要通过值传递大型的结构体。

kernel函数参数与device函数参数是不同的，一个是内存的访问与共享权限不同(read-only对比read-write，per-kernel对比per-thread)。

PTX 1.x版本只支持kernel函数参数，从2.0开始`.param`才支持device函数参数（需要`sm_20`及以上架构）。

【PS】PTX代码不应该对`.param`空间变量的相对位置或顺序做任何假设。(个人理解应该保持唯一的相对顺序)

#### 5.1.6.1.  Kernel Function Parameters
每个内核函数定义都包含一个可选的参数列表。这些参数是在`.param`状态空间中声明的可寻址只读变量。通过使用`ld.param`指令访问内核参数值。内核参数变量被grid内的所有线程共享。

内核参数的地址可以使用`mov`指令移动到寄存器中。结果地址在`.param`状态空间中，可以使用`ld.param`指令访问。

两个例子：
```
.entry foo ( .param .b32 N, .param .align 8 .b8 buffer[64] )
{
    .reg .u32 %n;
    .reg .f64 %d;
    ld.param.u32 %n, [N];
    ld.param.f64 %d, [buffer];
    ...
```

```
.entry bar ( .param .b32 len )
{
    .reg .u32 %ptr, %n;
    mov.u32 %ptr, len; // 寄存器%ptr指向len变量的地址
    ld.param.u32 %n, [%ptr]; //寄存器%n读取%ptr指针指向的值
```
【PS】:现阶段的应用中，不循序创建一个指向由kernel参数传入的常量内存的通用指针。(没试过，不太确定`cvta.const`指令是什么)

#### 5.1.6.2.  Kernel Function Parameter Attributes
kernel函数参数可以用可选的`.ptr`属性声明，可用来指示参数是指向内存的指针，也可表明指针所指向内存的状态空间和对齐方式。

#### 5.1.6.3.  Kernel Parameter Attribute: .ptr
`.ptr`语法：
```
.param .type .ptr .space .align N varname
.param .type .ptr .align N varname

.space = { .const, .global, .local, .shared };
```

其中`.space`和`.align`是可选的属性，`.space`缺失则默认是`.const, .global, .local, .shared`中的一种（基本属于未定义，所以一般还是不建议省略），`.align`缺失则默认按照4 byte对齐。

【PS】:`.ptr`、`.space`和`.align`之间不能有空格。

举个例子：
```
.entry foo ( .param .u32 param1,
            .param .u32 .ptr.global.align 16 param2,
            .param .u32 .ptr.const.align 8 param3,
            .param .u32 .ptr.align 16 param4 // generic address
            // pointer
) { .. }
```

#### 5.1.6.4.  Device Function Parameters
从PTX2.0开始扩展了device参数空间的使用，最常见的用法是不按照寄存器大小传值，如传入8 bytes大小的结构体参数。

举个例子：
```
// pass object of type struct { double d; int y; };
.func foo ( .reg .b32 N, .param .align 8 .b8 buffer[12] )
{
 .reg .f64 %d;
 .reg .s32 %y;
 ld.param.f64 %d, [buffer];
 ld.param.s32 %y, [buffer+8];
 ...
}

// 下面的片段来自kernel中对于device函数的调用
// struct { double d; int y; } mystruct; is flattened, passed to foo
 ...
 .reg .f64 dbl;
 .reg .s32 x;
 .param .align 8 .b8 mystruct; // 在local内存上声明结构体
 ...
 st.param.f64 [mystruct+0], dbl; // 结构体赋值
 st.param.s32 [mystruct+8], x;   // 结构体赋值
 call foo, (4, mystruct);        // device函数调用，传参
```

函数的输入参数可以使用`ld.param`进行读，返回值可以使用`st.param`进行写。

但是写input参数和读返回值都是不合法的。

除了按值传递结构外，当形式形参的地址在被调用的函数中被取时，还需要`.param`空间标注。

在PTX中，函数输入参数的地址可以使用`mov`指令移动到寄存器中。注意，如果需要，参数将被复制到堆栈中，因此地址将位于`.local`状态空间中，并通过`ld.local`和`st.local`指令进行访问。

不能使用`mov`来获取局部作用域的`.param`空间变量的地址。从PTX ISA 6.0版本开始，可以使用`mov`指令获取设备函数返回参数的地址。

举个例子：
```
// pass array of up to eight floating-point values in buffer
.func foo ( .param .b32 N, .param .b32 buffer[32] )
{
 .reg .u32 %n, %r;
 .reg .f32 %f;
 .reg .pred %p;
 ld.param.u32 %n, [N];
 // 注意此处，如果buffer实在foo的local-scope内部，那么是不能使用mov来获取地址的。
 mov.u32 %r, buffer; // forces buffer to .local state space
Loop:
 setp.eq.u32 %p, %n, 0;
@%p: bra Done;
 ld.local.f32 %f, [%r];
 ...
 add.u32 %r, %r, 4;
 sub.u32 %n, %n, 1;
 bra Loop;
Done:
 ...
}
```

### 5.1.7.  Shared State Space
`.shared`共享内存属于执行运算的CTA并且可以被同属一个cluster中的所有CTA的线程读写。

附加的子限定符`::cta`或`::cluster`可以在使用`.shared`的指令中指定状态空间，指示该地址是否属于正在执行的CTA或cluster中的任何CTA的共享内存。(即cluster共享，还是CTA内部共享)

`.shared::cta`的地址窗口也属于`.shared::cluster`的地址窗口。如果`.shared`状态空间中没有指定子限定符，则默认为`::cta`。例如，`ld.shared`等价于`ld.shared::cta`。

在`.shared`状态空间中声明的变量引用**当前CTA**中的内存地址。指令`mapa`给出了cluster中另一个CTA中**对应变量**的`.shared::cluster`地址。

共享内存通常有一些优化来支持共享。一个例子是广播，所有线程从同一个地址读取。另一种是从顺序线程的顺序访问。

### 5.1.8.  Texture State Space (deprecated)
弃用的纹理状态空间，不多赘述。

## 5.2.  Types
### 5.2.1.  Fundamental Types
在PTX中，基本类型反映了目标架构支持的原生数据类型。基本类型同时指定类型和大小。

寄存器变量总是一种基本类型，指令对这些类型进行操作。

基本类型如下：
![Table8](./images/table8.png)

大多数指令都有一个或多个类型说明符，用于完全指定指令的行为。操作数类型和大小将根据指令类型进行检查，以确保兼容性.

位大小相同的任何基本类型之间都是兼容的。

原则上，所有基本类型(除开predicate类型)可以只用位大小进行声明，但标明具体类型，可以提升可读性并且方便做类型检查。

### 5.2.2.  Restricted Use of Sub-Word Sizes
`.u8`、`.s8`和`.b8`被限制在`ld`、`st`和`cvt`指令中使用。

`.fp16`只能被用在与`fp32`和`fp64`的相互转化中，以及半精度浮点指令和纹理获取指令中。

`.fp16x2`只能被用在半精度浮点指令和纹理获取中。

为了方便起见，`ld`、`st`和`cvt`指令允许源操作数和目标数据操作数比指令类型的大小更宽。

例如，在加载、存储或转换为其他类型和大小时，8位或16位的值可能直接保存在32位或64位寄存器中。

### 5.2.3.  Alternate Floating-Point Data Formats
PTX中支持的基本浮点类型具有隐式的位表示，表示用于存储指数和尾数的位数。（也就是说对于浮点来说，有多种不同的位表示方规则）。

比如：IEEE 754的标准fp16的位组合规则是，1个符号位 + 5个指数位 + 10个精度位。简称位s1-e5-m10

在PTX中还额外支持如下特殊的半精度位组合：

- bf16
  - s1-e8-m7，寄存器中的`bf16`必须被声明为`.b16`。
- e4m3
  - s1-e4-m3，e4m3编码不支持infinity和Nan，被限制在0x7f和0xff。e4m3必须以pack的形式`e4m3x2`，并且必须被声明位`.b16`。
- e5m2
  - s1-e5-m2，同e4m3类似，也必须以pack的形式使用`e5m2x2`，并且必须被声明为`.b16`。
- ft32
  - 这是一种特殊的32位浮点，由矩阵乘和累加指令支持，范围与fp32相同，但是精度低一些（>=10bit），具体的内部布局由实现定义。PTX便于从`.fp32`到`.tf32`的转化，`.tf32`寄存器必须声明为`.b32`。

替代数据格式不能用作基本类型。它们被某些指令支持为源格式或目标格式。

## 5.3.  Texture Sampler and Surface Types
PTX中有一些内建的**不透明**类型来定义`texture`、`sampler`、`surface descriptor`变量。

这些类型的命名字段类似于结构体，但所有的信息如：布局、字段顺序、基址和总体大小都隐藏在PTX程序中，因此称为**不透明**。

这些不透明类型的使用有如下限制：

1. 变量定义在全局(module)作用域和内核参数列表中；
2. module-scope变量的静态初始化使用逗号隔开静态赋值表达式；
3. texture\sampler\surface的引用通过texture\surface的load\save指令完成`tex,suld,sust,sured`。
4. 通过查询指令检索指定成员的值；
5. 创建指向不透明变量的指针可以使用`mov`指令，如：`mov.u64 reg, opaque_var`。产生的指针可以从内存中读写，也可以通过参数传递给函数，还可以被texture\surface的读写查询指令所引用。
6. **不透明变量不能出现在初始化中，如：初始化一个指针指向不透明变量。**

【PS】:从PTX ISA 3.1版本开始支持使用指向不透明变量的指针间接访问texture\surface，需要目标架构`sm_20`及以上。

上述的三种内建的不透明类型是`.texref`、`.samplerref`和`.surfref`。

使用texture + sampler的时候，由两种操作模式可以选择:
1. 一种是`unified mode`，这种模式下，texture和sampler都用过单个`.texref`进行访问。
2. 另一种模式是`independent mode`，这种模式下，texture和sampler都有各自的句柄，允许他们分开定义再合并使用，在这种模式下`.texref`中关于sampler的定义将被忽略，因为会在`.samplerref`被定义。

下面两张表列出了在两种模式下面各种成员，这些成员及其值有具体的获取方法，在纹理`HW`类中定义，以及通过API查询。

- Unified Mode
![Table9](./images/table9.png)

- Independent Mode
![Table10](./images/table10.png)

### 5.3.1.  Texture and Surface Properties
上表中的`width`、`height`和`depth`表示texture\surface在每个维度的元素个数(更准确的说可以理解为像素pixel)。

其中每一个像素的属性可以由`channel_data_type`和`channel_order`来表示。、

OpenCL中的定义是被PTX支持的，所以可以参考OpenCL的定义如下：

![Table11](./images/table11.png)
![Table12](./images/table12.png)

### 5.3.2.  Sampler Properties
关于sampler的属性，代表的意义如下：

- `normalized_coords`：表示坐标是否归一化为[0.0, 1.0f]。如果没有被显式设置，则会在runtime阶段根据源码进行设置(也就是说还有可能默认被设为开启？？没试过，通常是默认非归一化的)
- `filter_mode`：表示如何基于坐标值计算texture读取的像素。
- ` addr_mode_{0,1,2}`： 定义了每个维度的寻址模式，该模式决定了每个维度如何处理out-of-range的坐标。
- `force_unnormalized_coords`: **在Independant Mode**下独有的属性，字面意思很好理解，会去将texture中的`normalized_coords`强行改写为`unnormalized_coords`当其被设置为`True`时，如果是`False`，那么就默认使用texture中的设置。
  - 【PS】:该属性被用在编译OpenCL to PTX的时候。

我们在声明这些不透明类型的时候，如果位于module-scope中，则需要使用`.global`状态空间；而如果位于kernel参数列表中，则需要使用`.param`状态空间。

举个例子：
```
.global .texref my_texture_name;
.global .samplerref my_sampler_name;
.global .surfref my_surface_name;
```

在`,.global`状态空间中，可以使用静态列表进行初始化，如：
```
.global .texref tex1;
.global .samplerref tsamp1 = { addr_mode_0 = clamp_to_border,
                               filter_mode = nearest
                             };
```

### 5.3.3.  Channel Data Type and Channel Order Fields
见5.3.1的表

## 5.4.  Variables



























