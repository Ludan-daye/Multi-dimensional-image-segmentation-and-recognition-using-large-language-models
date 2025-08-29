#!/bin/bash
# Training V4 智能启动脚本
# 集成自动配置、环境检测、智能参数调优

echo "🚀 Training V4 - 智能训练系统"
echo "=================================="
echo "🎯 核心任务: 图像分类 (<CLS>位置预测<CLS_X>)"
echo "📊 训练逻辑: 明确目标，避免理解错误"
echo ""

# 检查基础环境
check_environment() {
    echo "🔍 环境检查..."
    
    # Python检查
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 未安装"
        exit 1
    fi
    
    # CUDA检查
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA驱动未安装"
        exit 1
    fi
    
    # GPU状态
    echo "🎯 GPU状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while IFS=',' read -r name total used free; do
        name=$(echo $name | xargs)
        total=$(echo $total | xargs)
        used=$(echo $used | xargs)
        free=$(echo $free | xargs)
        
        used_percent=$((used * 100 / total))
        echo "  GPU: $name"
        echo "  内存: ${used}MB/${total}MB (${used_percent}%) | 空闲: ${free}MB"
        
        # 检查可用内存
        if [ $free -lt 4000 ]; then
            echo "  ⚠️ 警告: GPU内存不足4GB，可能影响训练"
        else
            echo "  ✅ GPU内存充足"
        fi
    done
    echo ""
}

# 检查Python依赖
check_dependencies() {
    echo "📦 检查Python依赖..."
    
    required_packages=("torch" "transformers" "numpy" "tqdm")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_packages+=($package)
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "❌ 缺少依赖包: ${missing_packages[*]}"
        echo "💡 安装命令: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    echo "✅ 所有依赖包已安装"
    echo ""
}

# 检查训练数据
check_training_data() {
    echo "📊 检查训练数据..."
    
    # 检查v3数据文件
    V3_DATA="../training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl"
    if [ ! -f "$V3_DATA" ]; then
        echo "❌ 找不到V3训练数据: $V3_DATA"
        echo "💡 请先完成training_v3的数据准备"
        exit 1
    fi
    
    # 检查词汇表
    V3_VOCAB="../training_v3/outputs/best_model_silent/vocab.json"
    if [ ! -f "$V3_VOCAB" ]; then
        echo "❌ 找不到词汇表: $V3_VOCAB"
        echo "💡 请确保training_v3已完成训练"
        exit 1
    fi
    
    # 统计数据信息
    DATA_SIZE=$(wc -l < "$V3_DATA")
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$V3_VOCAB'))))" 2>/dev/null || echo "未知")
    
    echo "✅ 训练数据检查通过"
    echo "  - 数据文件: $V3_DATA"
    echo "  - 样本数量: $DATA_SIZE"
    echo "  - 词汇表大小: $VOCAB_SIZE tokens"
    echo ""
}

# 智能配置建议
suggest_config() {
    echo "🧠 智能配置建议..."
    
    # 获取GPU内存
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_MEMORY_GB=$((GPU_MEMORY / 1024))
    
    echo "  🎯 基于GPU内存(${GPU_MEMORY_GB}GB)的建议配置:"
    
    if [ $GPU_MEMORY_GB -ge 80 ]; then
        echo "    - 模型规模: Large (1024d-16层)"
        echo "    - 批次大小: 16"
        echo "    - 学习率: 1e-4"
        SUGGESTED_CONFIG="large"
    elif [ $GPU_MEMORY_GB -ge 24 ]; then
        echo "    - 模型规模: Medium (768d-12层)"
        echo "    - 批次大小: 12" 
        echo "    - 学习率: 3e-4"
        SUGGESTED_CONFIG="medium"
    elif [ $GPU_MEMORY_GB -ge 12 ]; then
        echo "    - 模型规模: Small (512d-8层)"
        echo "    - 批次大小: 8"
        echo "    - 学习率: 5e-4"
        SUGGESTED_CONFIG="small"
    else
        echo "    - 模型规模: Tiny (384d-6层)"
        echo "    - 批次大小: 4"
        echo "    - 学习率: 5e-4"
        SUGGESTED_CONFIG="tiny"
    fi
    
    echo "  📊 预计训练时间: 2-4小时 (200 epochs)"
    echo ""
}

# 创建运行配置
create_run_config() {
    echo "⚙️ 创建运行配置..."
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_NAME="v4_${SUGGESTED_CONFIG}_${TIMESTAMP}"
    
    # 创建输出目录
    mkdir -p outputs/runs/$RUN_NAME
    mkdir -p logs
    
    # 生成配置文件
    cat > "config/run_config.json" << EOF
{
    "run_name": "$RUN_NAME",
    "suggested_config": "$SUGGESTED_CONFIG",
    "gpu_memory_gb": $GPU_MEMORY_GB,
    "training_start_time": "$(date -Iseconds)",
    "data_file": "$V3_DATA",
    "vocab_file": "$V3_VOCAB",
    "training_objective": "图像分类任务: <CLS>位置预测<CLS_X>分类token",
    "expected_behavior": "模型应该学习在<CLS>token位置预测正确的数字分类标签"
}
EOF

    echo "✅ 运行配置已创建: config/run_config.json"
    echo "  - 运行名称: $RUN_NAME"
    echo "  - 建议配置: $SUGGESTED_CONFIG"
    echo ""
}

# 启动训练
start_training() {
    echo "🚀 启动V4训练..."
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
    
    # 日志文件
    LOG_FILE="logs/v4_training_${TIMESTAMP}.log"
    PID_FILE="outputs/v4_training.pid"
    
    echo "📝 日志文件: $LOG_FILE"
    echo "🔧 进程文件: $PID_FILE"
    echo ""
    
    # 显示最终配置
    echo "📋 最终训练配置:"
    echo "  - 任务目标: 图像分类"
    echo "  - 训练逻辑: <CLS>位置预测<CLS_X>"
    echo "  - 模型规模: $SUGGESTED_CONFIG"
    echo "  - GPU内存: ${GPU_MEMORY_GB}GB"
    echo "  - 数据样本: $DATA_SIZE"
    echo "  - 预期结果: 准确率>20%"
    echo ""
    
    # 确认启动
    read -p "🤔 确认启动训练? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 训练已取消"
        exit 0
    fi
    
    echo "🚀 正在启动训练..."
    
    # 后台启动训练
    nohup python3 -u core/v4_trainer.py > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    
    sleep 3
    
    # 检查进程是否正常启动
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "✅ 训练成功启动!"
        echo "  - 进程ID: $TRAIN_PID"
        echo "  - 实时日志: tail -f $LOG_FILE"
        echo "  - 监控脚本: ./scripts/monitor_v4_training.sh"
        echo ""
        echo "🎉 V4训练已在后台运行，您可以安全关闭终端"
        echo "📊 预计完成时间: $(date -d '+4 hours' '+%H:%M')"
    else
        echo "❌ 训练启动失败"
        echo "💡 请检查日志: $LOG_FILE"
        exit 1
    fi
}

# 主流程
main() {
    # 进入脚本目录
    cd "$(dirname "$0")/.."
    
    # 执行检查流程
    check_environment
    check_dependencies
    check_training_data
    suggest_config
    create_run_config
    start_training
}

# 执行主流程
main "$@"
