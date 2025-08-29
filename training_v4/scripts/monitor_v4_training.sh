#!/bin/bash
# Training V4 智能监控脚本
# 实时监控训练状态、GPU使用率、准确率趋势

clear
echo "📊 Training V4 智能监控系统"
echo "==============================="
echo "🎯 监控目标: 图像分类训练"
echo "📈 关键指标: <CLS>位置分类准确率"
echo ""

# 检查训练进程
check_training_process() {
    local pid_file="outputs/v4_training.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ 训练进程运行中 (PID: $pid)"
            
            # 获取进程信息
            local cpu_usage=$(ps -p $pid -o %cpu --no-headers | xargs)
            local memory_usage=$(ps -p $pid -o %mem --no-headers | xargs)
            local start_time=$(ps -p $pid -o lstart --no-headers)
            
            echo "  - CPU使用率: ${cpu_usage}%"
            echo "  - 内存使用率: ${memory_usage}%"
            echo "  - 启动时间: $start_time"
        else
            echo "❌ 训练进程已停止"
            rm -f "$pid_file"
        fi
    else
        echo "❌ 没有找到运行的训练进程"
    fi
    echo ""
}

# GPU状态监控
monitor_gpu_status() {
    echo "🎯 GPU状态监控:"
    
    if command -v nvidia-smi &> /dev/null; then
        # GPU基本信息
        nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r name temp power mem_used mem_total util; do
            name=$(echo $name | xargs)
            temp=$(echo $temp | xargs)
            power=$(echo $power | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            util=$(echo $util | xargs)
            
            mem_percent=$((mem_used * 100 / mem_total))
            mem_free=$((mem_total - mem_used))
            
            echo "  GPU: $name"
            echo "  温度: ${temp}°C | 功耗: ${power}W | 利用率: ${util}%"
            echo "  显存: ${mem_used}MB/${mem_total}MB (${mem_percent}%) | 空闲: ${mem_free}MB"
            
            # 状态指示
            if [ $util -gt 80 ]; then
                echo "  状态: 🔥 高强度训练中"
            elif [ $util -gt 30 ]; then
                echo "  状态: ⚡ 正常训练中"
            else
                echo "  状态: 💤 GPU利用率较低"
            fi
        done
    else
        echo "  ⚠️ nvidia-smi 不可用"
    fi
    echo ""
}

# 训练进度监控
monitor_training_progress() {
    echo "📈 训练进度监控:"
    
    # 检查运行配置
    local config_file="config/run_config.json"
    if [ -f "$config_file" ]; then
        if command -v jq &> /dev/null; then
            local run_name=$(jq -r '.run_name' "$config_file")
            local config_type=$(jq -r '.suggested_config' "$config_file")
            local start_time=$(jq -r '.training_start_time' "$config_file")
            
            echo "  运行名称: $run_name"
            echo "  配置类型: $config_type"
            echo "  开始时间: $start_time"
        fi
    fi
    
    # 检查训练历史文件
    local progress_files=(
        "outputs/checkpoints/training_progress.json"
        "outputs/v4_training_progress.json"
    )
    
    local progress_found=false
    for progress_file in "${progress_files[@]}"; do
        if [ -f "$progress_file" ]; then
            progress_found=true
            echo "  📊 进度文件: $progress_file"
            
            if command -v jq &> /dev/null; then
                # 解析训练进度
                local current_epoch=$(jq -r '.current_epoch // 0' "$progress_file" 2>/dev/null)
                local total_epochs=$(jq -r '.total_epochs // 200' "$progress_file" 2>/dev/null)
                local current_accuracy=$(jq -r '.current_accuracy // 0' "$progress_file" 2>/dev/null)
                local best_accuracy=$(jq -r '.best_accuracy // 0' "$progress_file" 2>/dev/null)
                local current_loss=$(jq -r '.current_loss // 0' "$progress_file" 2>/dev/null)
                
                # 计算进度百分比
                local progress_percent=0
                if [ "$total_epochs" -gt 0 ]; then
                    progress_percent=$((current_epoch * 100 / total_epochs))
                fi
                
                echo "  轮次: ${current_epoch}/${total_epochs} (${progress_percent}%)"
                echo "  当前准确率: ${current_accuracy}"
                echo "  最佳准确率: ${best_accuracy}"
                echo "  当前损失: ${current_loss}"
                
                # 进度条
                local bar_length=40
                local filled_length=$((progress_percent * bar_length / 100))
                local bar=""
                for ((i=0; i<filled_length; i++)); do bar+="█"; done
                for ((i=filled_length; i<bar_length; i++)); do bar+="░"; done
                echo "  进度: [${bar}] ${progress_percent}%"
                
                # 状态评估
                if [ $(echo "$current_accuracy > 0.20" | bc -l) -eq 1 ]; then
                    echo "  状态: 🎉 训练效果优秀!"
                elif [ $(echo "$current_accuracy > 0.15" | bc -l) -eq 1 ]; then
                    echo "  状态: ✅ 训练效果良好"
                elif [ $(echo "$current_accuracy > 0.10" | bc -l) -eq 1 ]; then
                    echo "  状态: 📈 训练有进展"
                elif [ "$current_epoch" -gt 20 ]; then
                    echo "  状态: ⚠️ 准确率较低，需要检查"
                else
                    echo "  状态: 🔄 训练初期"
                fi
            fi
            break
        fi
    done
    
    if [ "$progress_found" = false ]; then
        echo "  ⚠️ 未找到训练进度文件"
    fi
    echo ""
}

# 训练目标验证
validate_training_objective() {
    echo "🎯 训练目标验证:"
    echo "  主要目标: 图像分类任务"
    echo "  核心逻辑: 在<CLS>token位置预测正确的<CLS_X>分类token"
    echo "  成功标准: 分类准确率 > 15% (超越v3的最佳结果)"
    echo "  数据要求: CLS token识别率 > 95%"
    echo ""
}

# 最新日志显示
show_recent_logs() {
    echo "📝 最新训练日志 (最后20行):"
    echo "────────────────────────────────────"
    
    # 查找最新的日志文件
    local log_pattern="logs/v4_training_*.log"
    local latest_log=$(ls -t $log_pattern 2>/dev/null | head -n1)
    
    if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
        tail -n 20 "$latest_log" | while IFS= read -r line; do
            # 高亮重要信息
            if [[ $line == *"Epoch"* ]] && [[ $line == *"Acc:"* ]]; then
                echo "📊 $line"
            elif [[ $line == *"ERROR"* ]] || [[ $line == *"❌"* ]]; then
                echo "🚨 $line"
            elif [[ $line == *"WARNING"* ]] || [[ $line == *"⚠️"* ]]; then
                echo "⚠️  $line"
            elif [[ $line == *"✅"* ]] || [[ $line == *"best"* ]]; then
                echo "🎉 $line"
            else
                echo "   $line"
            fi
        done
    else
        echo "  ⚠️ 没有找到日志文件"
        echo "  💡 可能的日志位置: logs/v4_training_*.log"
    fi
    echo "────────────────────────────────────"
    echo ""
}

# 预警系统
check_alerts() {
    echo "🚨 智能预警系统:"
    
    local alerts_count=0
    
    # 检查GPU温度
    if command -v nvidia-smi &> /dev/null; then
        local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1)
        if [ "$gpu_temp" -gt 85 ]; then
            echo "  🔥 警告: GPU温度过高 (${gpu_temp}°C)"
            alerts_count=$((alerts_count + 1))
        fi
    fi
    
    # 检查磁盘空间
    local disk_usage=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        echo "  💾 警告: 磁盘空间不足 (${disk_usage}%)"
        alerts_count=$((alerts_count + 1))
    fi
    
    # 检查训练是否卡住
    local latest_log=$(ls -t logs/v4_training_*.log 2>/dev/null | head -n1)
    if [ -n "$latest_log" ]; then
        local last_update=$(stat -c %Y "$latest_log" 2>/dev/null || echo 0)
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_update))
        
        if [ $time_diff -gt 600 ]; then  # 10分钟没有更新
            echo "  ⏰ 警告: 训练日志超过10分钟没有更新"
            alerts_count=$((alerts_count + 1))
        fi
    fi
    
    if [ $alerts_count -eq 0 ]; then
        echo "  ✅ 所有系统正常运行"
    else
        echo "  📊 发现 $alerts_count 个需要注意的问题"
    fi
    echo ""
}

# 操作提示
show_operations() {
    echo "🛠️ 可用操作:"
    echo "  - 停止训练: kill $(cat outputs/v4_training.pid 2>/dev/null || echo 'N/A')"
    echo "  - 查看完整日志: tail -f logs/v4_training_*.log"
    echo "  - 检查检查点: ls -la outputs/checkpoints/"
    echo "  - 重启监控: $0"
    echo ""
    echo "🔄 自动刷新间隔: 30秒"
    echo "⏸️  按 Ctrl+C 退出监控"
}

# 主监控循环
main_monitor() {
    while true; do
        clear
        echo "📊 Training V4 智能监控系统 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"
        
        check_training_process
        monitor_gpu_status
        monitor_training_progress
        validate_training_objective
        show_recent_logs
        check_alerts
        show_operations
        
        echo "正在等待下次刷新..."
        sleep 30
    done
}

# 主函数
main() {
    # 检查依赖
    if ! command -v bc &> /dev/null; then
        echo "⚠️ 建议安装 bc 以获得更好的数值计算: sudo apt install bc"
    fi
    
    if ! command -v jq &> /dev/null; then
        echo "⚠️ 建议安装 jq 以获得更好的JSON解析: sudo apt install jq"
    fi
    
    # 进入脚本目录
    cd "$(dirname "$0")/.."
    
    # 开始监控
    main_monitor
}

# 执行主函数
main "$@"
