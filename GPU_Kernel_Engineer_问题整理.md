# jinboom：AI Infra / HPC 面试问题清单

> 来源：[知乎回答《为什么 AI Infra 里还在大量招聘 GPU Kernel Engineer？》](https://www.zhihu.com/question/1998102573333385508/answer/2005326754357725076)  
> 答主：jinboom  
> 提取依据：用户保存的 PDF 第 1～4 页（截至“OriginDL 能帮你什么”部分）  
> 整理规则：仅提取 jinboom 正文中的问题和复习要点；为方便复习，将部分陈述式要点统一改写为问句，但不补充答案。PDF 第 5～6 页的其他答主内容未收录。

## 1. 芯片与算力

1. GPU 和 CPU 真正的区别在哪里？
2. SM、Warp、SIMD、SIMT 分别是什么？
3. Warp 分化会带来哪些性能问题？
4. 训练速度慢时，通常会受哪些硬件指标限制，例如 FLOPS、显存带宽和 PCIe 带宽？
5. MFU（Model FLOPS Utilization）是什么概念？
6. 如何通过 MFU 判断算力有没有被充分利用？

## 2. CUDA 与高性能计算

### 2.1 CUDA 编程模型

1. Grid、Block、Thread 三层执行结构为什么要划分得这么细？
2. Block 为什么这样设计？
3. `__global__`、`__device__`、`__host__`、`__shared__`、`__constant__`、`__restrict__`、`__managed__` 等修饰符分别有什么作用和典型使用场景？

### 2.2 执行模型与 GPU 架构

4. Block 为什么不能跨 SM 拆分？
5. Block 是如何被调度到 SM 上执行的？
6. Warp 是如何执行的？
7. SIMD 和 SIMT 有什么本质区别？

### 2.3 内存层次与性能优化

8. GPU 中的寄存器、Local Memory、Shared Memory、Global Memory、Constant Memory 和 Texture Memory，速度层次如何？
9. 上述各种 GPU 内存分别适合什么场景？
10. 对齐与合并访问（Coalesced Access）是什么？
11. 访存不合并为什么会明显降低性能？

### 2.4 流与并发

12. CUDA Stream 是做什么的？
13. 如何使用 CUDA Stream 重叠计算与数据拷贝？
14. Hyper-Q 解决的是哪类“硬件很闲，但任务排不上”的问题？

### 2.5 调试与性能分析

15. 使用 Nsight、Nsight Compute（ncu）和 nvprof 时，通常要看哪些核心指标？
16. 拿到一个运行缓慢的 Kernel，应按照什么顺序排查访存、占用率、指令和并发等问题？
17. CUDA-GDB 与普通 C++ 调试相比，有哪些与线程维度和设备内存相关的注意事项？

## 3. 多机多卡互联与拓扑

1. NVLink 相比 PCIe 强在哪里？
2. NVLink 与 PCIe 的带宽和延迟大致有多大差异？
3. NVSwitch 如何把多块 GPU 组成全互联系统？
4. 常见的八卡 GPU 拓扑是什么样的？
5. 串行、并行和偏 HPC 的常见八卡服务器拓扑各有什么取舍？
6. 如果选择一台用于训练大模型的服务器，应关注哪些因素，例如直连链路、CPU/GPU 亲和性和 IB 网卡所在的 NUMA 节点？
7. NUMA 是什么？
8. 为什么绑核、绑卡会对性能产生很大影响？
9. 如何在 Linux 中查看设备的 NUMA 信息？
10. 亲和性配置错误会导致哪些明显的性能抖动？

## 4. GPUDirect

1. GPUDirect Storage（GDS）、GPUDirect P2P 和 GPUDirect RDMA（GDR）分别解决什么问题？
2. GPUDirect P2P 和 GPUDirect RDMA 有什么区别？
3. 网卡如何直接读写 GPU 显存？
4. 网卡直接访问 GPU 显存时，经历了哪些映射和内存注册过程？
5. GPU 如何访问网卡寄存器、队列等通信资源？
6. GPU 如何提交通信请求并与网卡同步？
7. 所谓的“门铃机制”是怎么回事？
8. 锁页内存（Pinned Memory）、零拷贝内存、UVA 和统一内存（UM）分别适合什么场景？
9. 为什么 Pinned Memory 通常能提供更高带宽？
10. Pinned Memory 是否可能导致系统崩溃，原因是什么？

## 5. RDMA 与 InfiniBand

### 5.1 基础与优势

1. RDMA 到底是什么？
2. RDMA 绕开传统操作系统内核的数据路径后，能带来哪些直接收益？
3. SEND/RECV、READ/WRITE 以及带立即数的操作分别适合什么场景？

### 5.2 协议、设备与资源模型

4. RDMA、InfiniBand、IBoE、RoCE、iWARP、IB 卡和 IB 驱动之间是什么关系？
5. RoCE v1 和 RoCE v2 在协议及性能上有哪些主要差异？
6. QP、CQ、MR、MW、PD、AH 分别抽象了什么资源？
7. 为什么 RDMA 必须执行内存注册？
8. 注册 GPU 显存时会多出哪些限制？

### 5.3 性能调优与排障

9. 优先优化小包延迟时，通常会采用哪些手段，例如批量 Post、Inline 和减少信号操作？
10. 优化带宽时，为什么要重点关注 MTU、QP 数量、NUMA 和 PCIe 抖动？
11. RDMA 延迟抖动与 CPU 隔离、中断绑定及亲和性配置有什么关系？
12. 从零部署一套 IB/RDMA 环境，需要安装和配置什么？
13. 可以使用哪些工具自检 IB/RDMA 环境？
14. 遇到 `PORT DOWN`、端口无法启动或连接失败时，应如何排查？

## 6. 集合通信与 NCCL

### 6.1 集合通信算法

1. AllReduce、AllGather、ReduceScatter 和 Broadcast 分别做什么？
2. Ring、Tree 和 Bruck 等实现各有什么优缺点？
3. Ring、Tree 和 Bruck 中，哪些延迟更好，哪些带宽利用率更高？

### 6.2 NCCL 的设计与协议

4. NCCL 如何利用 NVLink、PCIe 和 InfiniBand 等实际硬件拓扑建图？
5. NCCL 中的 Channel 是什么？
6. NCCL 为什么要把通信拆成多个 Channel？
7. Simple、LL 和 LL128 三种协议分别适合什么场景？
8. Simple、LL 和 LL128 对延迟及带宽分别有什么影响？
9. LL128 为什么使用更大的块？
10. LL128 采用更大块的取舍是什么？

### 6.3 实战问题

11. 多机多卡训练时，AllReduce 卡住或速度忽快忽慢，应从拓扑、环境变量、`NCCL_DEBUG` 日志、IB 计数器和交换机等哪些方向开始排查？
12. 千卡规模的集群训练中，有哪些常见问题可以提前规避？

## 7. 训练与推理系统

### 7.1 参数量与显存

1. 10 亿（1B）参数大约需要占用多少显存？
2. 训练和推理阶段的显存主要分别消耗在哪里？
3. 为什么训练时激活值会占用大量显存？

### 7.2 并行与分布式训练

4. 数据并行（DP/DDP）、模型并行（张量并行、流水线并行）和专家并行（MoE）的核心思想分别是什么？
5. 各类并行策略分别适合什么场景？
6. ZeRO 各个 Stage 分别减少了参数、梯度和优化器状态中的哪些显存占用？

### 7.3 指标与瓶颈定位

7. MFU、GPU 利用率和吞吐量（Tokens/s）之间是什么关系？
8. 如何通过监控判断当前瓶颈是算力未充分利用，还是网络或 I/O 拖慢了系统？

### 7.4 推理优化

9. 推理阶段常见的优化手段有哪些，例如 KV Cache、Batching、并行策略和 Runtime 优化？

## 8. 工程实践

1. 是否从零到一实现过 CUDA Kernel 并完成性能调优？
2. 对 CUDA Kernel 进行性能优化时，分析和修改的完整路线是什么？
3. 是否亲手处理过 RDMA、NCCL 或 NVLink 相关的 Bug？
4. 能否举例说明如何发现通信问题，并一步步缩小排查范围？
5. 给定一台八卡机器，如何尽量榨干训练性能？
6. 日常分析性能瓶颈时会使用哪些工具，例如 Nsight、`nvidia-smi dmon`、IB 计数器、perf、Ftrace 和 eBPF？

## 9. 可结合 OriginDL 项目回答的问题

1. 某个算子的 CUDA Kernel 是如何设计的？
2. Kernel 的 Block 和 Grid 应如何配置？
3. Shared Memory 和 Global Memory 应如何配合使用？
4. 高层 API 调用一次 `forward` 时，底层具体触发了哪些 Kernel？
5. 训练或推理链路中的哪些位置使用了 Stream、Event 和异步拷贝？
6. 训练或推理链路中的哪些环节容易出现性能或显存问题？

## 汇总

本文共整理出 **90 个问题**，覆盖九个主题：

1. 芯片与算力；
2. CUDA 与高性能计算；
3. 多机多卡互联与拓扑；
4. GPUDirect；
5. RDMA 与 InfiniBand；
6. 集合通信与 NCCL；
7. 训练与推理系统；
8. 工程实践；
9. OriginDL 项目实践。
