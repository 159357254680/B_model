@echo off
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ==============================================
    echo   未检测到 Docker，请先安装：
    echo   https://www.docker.com/products/docker-desktop/
    echo ==============================================
    exit /b 1
)

if "%1"=="" (
    echo 用法: run_docker.bat [mock^|full^|interactive]
    echo   mock        - 模拟数据快速测试
    echo   full        - IMDB 真实数据 + 全链路分析
    echo   interactive - 进入容器手动操作
    exit /b 1
)

echo === 构建 / 更新镜像 ===
docker compose build

if "%1"=="mock" (
    echo === Docker: mock 模式 ===
    docker compose run --rm b_model python train.py --mock
) else if "%1"=="full" (
    echo === Docker: 完整流程 ===
    docker compose run --rm b_model bash -c "python a_preprocess.py && python train.py"
) else if "%1"=="interactive" (
    echo === Docker: 交互模式 ===
    docker compose run --rm b_model bash
) else (
    echo 未知模式: %1
    echo 用法: run_docker.bat [mock^|full^|interactive]
)
