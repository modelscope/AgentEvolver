#!/usr/bin/env bash
set -e

echo "Installing AgentEvolver environment..."
echo

# ---- Step 1. Check Conda installation ----
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not found in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---- Step 2. Ask user for environment name ----
ENV_NAME="agentevolver"
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists. If you need to reinstall it, please delete the existing environment first."
    exit 1
fi

# ---- Step 3. Create new environment ----
echo
echo "📦 Creating environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python=3.11

# ---- Step 4. Activate environment ----
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- Step 5. Install CUDA toolkit ----
echo
echo "🚀 Installing CUDA toolkit ..."
conda install -y -c nvidia cuda-toolkit

# ---- Step 6. Install Python dependencies ----
if [[ ! -f requirements.txt ]]; then
    echo "⚠️  No requirements.txt found in current directory. Please check your working directory."
    exit 1
else
    echo
    echo "📥 Installing packages from requirements.txt ..."
    pip install -r requirements.txt
fi

# ---- Step 7. Install FlashAttention packages ----
echo
echo "⚙️  Installing flash-attn libraries ..."
pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn --no-build-isolation

# ---- Step 8. Finish ----
echo
echo "✅ Installation complete!"
echo "Environment '$ENV_NAME' is ready for your AgentEvolver! Please follow the rest instructions to start training."
echo
echo "To activate it later, run:"
echo "  conda activate $ENV_NAME"
echo
