#!/bin/bash
# setup_env.sh - 卫星边缘计算ISAC项目一键环境配置脚本

set -e  # 任何命令失败则退出脚本

echo -e "\033[1;36m[卫星边缘计算ISAC] 环境配置启动...\033[0m"
echo "================================================"

# 1. 安装系统级依赖
echo -e "\033[1;33m[步骤1/5] 安装系统依赖...\033[0m"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libgsl-dev \
    libxml2-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    qt5-default \
    ccache  # 加速编译

# 2. 配置Conda环境
echo -e "\033[1;33m\n[步骤2/5] 设置Conda环境...\033[0m"
if ! command -v conda &> /dev/null; then
    echo "检测到未安装Miniconda，正在下载安装..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    source $HOME/miniconda/etc/profile.d/conda.sh
fi

# 创建专用环境
conda create -n isac python=3.11 -y
conda activate isac

# 3. 安装Python依赖
echo -e "\033[1;33m\n[步骤3/5] 安装Python依赖...\033[0m"
pip install --upgrade pip
pip install -q torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -r requirements.txt

# 4. 编译安装ns-3
echo -e "\033[1;33m\n[步骤4/5] 编译ns-3（启用ccache加速）...\033[0m"
if [ ! -d "ns-3-dev" ]; then
    git clone https://gitlab.com/nsnam/ns-3-dev.git --depth=1
fi

cd ns-3-dev

# 配置5G/LTE模块
./ns3 configure --enable-examples \
                --enable-tests \
                --enable-sudo \
                --enable-modules='lte,mmwave,nr' \
                --build-profile=optimized \
                --out=build/optimized \
                --ccache

# 并行编译（使用所有CPU核心）
CPU_COUNT=$(nproc)
echo -e "\033[1;34m使用${CPU_COUNT}线程进行编译...\033[0m"
./ns3 build -j$CPU_COUNT

# 安装Python绑定
./ns3 build bindings

# 5. 验证安装
echo -e "\033[1;33m\n[步骤5/5] 验证环境...\033[0m"
cd ..

# 运行单元测试
echo -e "\033[1;32m\n✅ 运行单元测试...\033[0m"
pytest tests/ -q

# 显示关键工具版本
echo -e "\n\033[1;32m✅ 环境配置完成！验证版本信息:\033[0m"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
./ns-3-dev/ns3 --version | head -n 1

echo -e "\n\033[1;36m================================================"
echo "🚀 环境准备就绪！使用以下命令激活:"
echo "   conda activate isac"
echo "   export PYTHONPATH=\$PWD/ns-3-dev/build/optimized/bindings/python:\$PYTHONPATH"
echo "================================================\033[0m"

# 生成环境变量配置脚本
cat > activate_env.sh << 'EOF'
#!/bin/bash
conda activate isac
export PYTHONPATH=$PWD/ns-3-dev/build/optimized/bindings/python:$PYTHONPATH
echo "ISAC环境已激活！"
EOF

chmod +x activate_env.sh
echo -e "\033[1;33m\⚠️ 注意: 每次启动新终端时运行: source activate_env.sh\033[0m"