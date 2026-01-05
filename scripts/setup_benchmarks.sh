set -e  # Exit immediately if a command exits with a non-zero status

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="$PROJECT_ROOT/benchmarks"
SETUP_DIR="$PROJECT_ROOT/setup"

echo -e "${GREEN}🚀 Setting up HiSA benchmarks...${NC}\n"

# Create benchmarks directory
mkdir -p "$BENCHMARK_DIR"
cd "$BENCHMARK_DIR"

# Clone Spider2-V
echo -e "${GREEN}📦 Setting up Spider2-V...${NC}"
if [ -d "Spider2-V/.git" ]; then
    echo -e "${YELLOW}Spider2-V already exists, skipping clone${NC}"
else
    git clone https://github.com/xlang-ai/Spider2-V.git
    echo "Spider2-V cloned successfully"
fi

# Apply HiSA modifications to Spider2-V
echo -e "${GREEN}🔧 Applying HiSA modifications to Spider2-V...${NC}"
if [ -d "$SETUP_DIR/Spider2-V" ]; then
    cp -rf "$SETUP_DIR/Spider2-V/"* Spider2-V/
    echo -e "${GREEN}✓ Spider2-V modifications applied${NC}"
else
    echo -e "${RED}✗ Warning: $SETUP_DIR/Spider2-V not found${NC}"
fi

# Clone OSWorld
echo -e "\n${GREEN}📦 Setting up OSWorld...${NC}"
if [ -d "OSWorld/.git" ]; then
    echo -e "${YELLOW}OSWorld already exists, skipping clone${NC}"
else
    git clone https://github.com/xlang-ai/OSWorld.git
    echo "OSWorld cloned successfully"
fi

# Apply HiSA modifications to OSWorld
echo -e "${GREEN}🔧 Applying HiSA modifications to OSWorld...${NC}"
if [ -d "$SETUP_DIR/OSWorld" ]; then
    cp -rf "$SETUP_DIR/OSWorld/"* OSWorld/
    echo -e "${GREEN}✓ OSWorld modifications applied${NC}"
else
    echo -e "${RED}✗ Warning: $SETUP_DIR/OSWorld not found${NC}"
fi

echo -e "\n${GREEN}✅ Benchmark setup complete!${NC}"
echo -e "\n${YELLOW}Modified files:${NC}"
echo "- Spider2-V: Files from setup/Spider2-V have been copied"
echo "- OSWorld: Files from setup/OSWorld have been copied"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Follow Spider2-V setup: cd benchmarks/Spider2-V && follow their README"
echo "2. Follow OSWorld setup: cd benchmarks/OSWorld && follow their README"
echo "3. Configure VM snapshots as described in HiSA README"