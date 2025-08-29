#!/bin/bash
# Training V4 智能启动脚本 - 修复版
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
        
        if [ $free -lt 4000 ]; then
            echo "  ⚠️ 警告: GPU内存不足4GB，可能影响训练"
        else
            echo "  ✅ GPU内存充足"
        fi
    done
    echo ""
}

# 检查训练数据 - 修复版
check_training_data() {
    echo "📊 检查训练数据..."
    
    # 检查v3数据文件
    V3_DATA="../training_v3/generated_sequences_super_enhanced/sequences_labels_fixed_v2_maxlen1024.jsonl"
    if [ ! -f "$V3_DATA" ]; then
        echo "❌ 找不到V3训练数据: $V3_DATA"
        exit 1
    fi
    
    # 检查词汇表 (多个可能位置) - 修复版
    echo "🔍 搜索词汇表文件..."
    V3_VOCAB_PATHS=(
        "../training_v3/outputs/best_model_fixed/vocab.json"
        "../training_v3/generated_sequences_super_enhanced/vocab.json"
        "../training_v3/outputs/best_model_silent/vocab.json"
    )
    
    V3_VOCAB=""
    for vocab_path in "${V3_VOCAB_PATHS[@]}"; do
        if [ -f "$vocab_path" ]; then
            V3_VOCAB="$vocab_path"
            echo "✅ 找到词汇表: $vocab_path"
            break
        fi
    done
    
    if [ -z "$V3_VOCAB" ]; then
        echo "💡 V4将使用默认词汇表配置"
        V3_VOCAB="默认词汇表"
        VOCAB_SIZE="519 (默认)"
    else
        VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$V3_VOCAB'))))" 2>/dev/null || echo "未知")
    fi
    
    DATA_SIZE=$(wc -l < "$V3_DATA")
    
    echo "✅ 训练数据检查完成"
    echo "  - 数据文件: $V3_DATA"
    echo "  - 样本数量: $DATA_SIZE"
    echo "  - 词汇表: $V3_VOCAB"
    echo "  - 词汇表大小: $VOCAB_SIZE tokens"
    echo ""
}

# 启动训练
start_training() {
    echo "🚀 现在可以启动V4训练了！"
    echo "💡 使用命令: python3 core/v4_trainer.py"
}

# 主流程
main() {
    cd "$(dirname "$0")/.."
    check_environment
    check_training_data
    start_training
}

main "$@"
