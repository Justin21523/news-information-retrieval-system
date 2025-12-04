#!/bin/bash
# Faceted Search 快速啟動腳本

echo "========================================="
echo "   Faceted Search 快速測試"
echo "========================================="
echo ""

# 停止現有 Flask
echo "⏸️  停止現有 Flask..."
pkill -f "python app_simple.py" 2>/dev/null
sleep 2

# 啟動 Flask
echo "🚀 啟動 Flask 應用..."
source activate ai_env
nohup python app_simple.py > /tmp/flask_app.log 2>&1 &
FLASK_PID=$!

echo ""
echo "✅ Flask 已啟動！"
echo "   Process ID: $FLASK_PID"
echo ""
echo "========================================="
echo "   系統資訊"
echo "========================================="
echo ""
echo "URL: http://localhost:5000"
echo "Log: tail -f /tmp/flask_app.log"
echo""

echo "⏳ 等待 Flask 初始化 (5秒)..."
sleep 5

# 檢查 Flask 狀態
if ps -p $FLASK_PID > /dev/null 2>&1; then
    echo "✅ Flask 正在運行！"
    echo ""
    echo "========================================="
    echo "   已知問題與說明"
    echo "========================================="
    echo ""
    echo "📌 當前問題："
    echo "  1. Facet 面板在上方（應在左側）"
    echo "  2. 文檔缺少摘要內容"
    echo "  3. 點擊文檔後可能看不到詳情"
    echo "  4. 篩選功能可能無反應"
    echo ""
    echo "🔍 調試步驟："
    echo "  1. 打開瀏覽器訪問 http://localhost:5000"
    echo "  2. 按 F12 開啟開發者工具"
    echo "  3. 切換到 Console 頁籤"
    echo "  4. 執行搜索並觀察錯誤訊息"
    echo ""
    echo "📋 API 測試："
    echo "  curl -X POST http://localhost:5000/api/search \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\":\"政治\",\"model\":\"bm25\",\"top_k\":10}'"
    echo ""
else
    echo "❌ Flask 啟動失敗！"
    echo "請查看日誌: tail -f /tmp/flask_app.log"
    exit 1
fi
