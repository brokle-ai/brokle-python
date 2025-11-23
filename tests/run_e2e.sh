#!/bin/bash
# ============================================================================
# Quick E2E Test Runner
# ============================================================================
# This script sets up and runs the E2E test suite with sensible defaults
#
# Usage:
#   ./tests/run_e2e.sh                    # Use defaults
#   ./tests/run_e2e.sh --install-deps     # Install dependencies first
#   OPENAI_API_KEY=sk-... ./tests/run_e2e.sh  # Include OpenAI test
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

print_banner() {
    echo ""
    echo "🚀════════════════════════════════════════════════════════════════════🚀"
    echo "    $1"
    echo "🚀════════════════════════════════════════════════════════════════════🚀"
    echo ""
}

# ============================================================================
# Configuration
# ============================================================================

# Default values
export BROKLE_API_KEY="${BROKLE_API_KEY:-bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK}"
export BROKLE_BASE_URL="${BROKLE_BASE_URL:-http://localhost:8080}"
export CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-localhost}"
export CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-8123}"  # HTTP port for clickhouse-connect
export CLICKHOUSE_USER="${CLICKHOUSE_USER:-brokle}"
export CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-brokle_password}"
export CLICKHOUSE_DATABASE="${CLICKHOUSE_DATABASE:-default}"

# ============================================================================
# Install Dependencies (if --install-deps flag)
# ============================================================================

if [[ "$1" == "--install-deps" ]]; then
    print_banner "Installing Test Dependencies"

    print_info "Installing clickhouse-connect..."
    pip install clickhouse-connect

    print_info "Installing openai (optional)..."
    pip install openai || print_warning "OpenAI install failed (optional dependency)"

    print_success "Dependencies installed"
    echo ""
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_banner "Pre-flight Checks"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

# Check if clickhouse-connect is installed
if ! python3 -c "import clickhouse_connect" 2>/dev/null; then
    print_error "clickhouse-connect not installed"
    print_info "Install with: pip install clickhouse-connect"
    print_info "Or run: ./tests/run_e2e.sh --install-deps"
    exit 1
fi
print_success "clickhouse-connect installed"

# Check backend
print_info "Checking backend at ${BROKLE_BASE_URL}..."
if curl -s "${BROKLE_BASE_URL}/health" > /dev/null 2>&1; then
    print_success "Backend is running"
else
    print_error "Backend not running at ${BROKLE_BASE_URL}"
    print_info "Start with: make dev-server"
    exit 1
fi

# Check ClickHouse
print_info "Checking ClickHouse at ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}..."
if nc -z "${CLICKHOUSE_HOST}" "${CLICKHOUSE_PORT}" 2>/dev/null; then
    print_success "ClickHouse is running"
else
    print_error "ClickHouse not running at ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"
    print_info "Start with: docker-compose up clickhouse"
    exit 1
fi

# Check API key
if [[ -z "${BROKLE_API_KEY}" ]]; then
    print_error "BROKLE_API_KEY not set"
    print_info "Set with: export BROKLE_API_KEY='bk_...'"
    exit 1
fi
print_success "API key configured (${BROKLE_API_KEY:0:10}...)"

# Check OpenAI key (optional)
if [[ -n "${OPENAI_API_KEY}" ]]; then
    print_success "OpenAI API key configured (Test 8 will run)"
else
    print_warning "OpenAI API key not set (Test 8 will be skipped)"
    print_info "Set with: export OPENAI_API_KEY='sk-...'"
fi

echo ""

# ============================================================================
# Run Tests
# ============================================================================

print_banner "Running E2E Tests"

# Navigate to SDK directory
cd "$(dirname "$0")/.."

# Run the test
print_info "Executing test_e2e_clickhouse.py..."
echo ""

if python3 tests/test_e2e_clickhouse.py; then
    print_banner "🎉 All Tests Passed!"
    exit 0
else
    print_banner "❌ Some Tests Failed"
    print_info "Check the output above for details"
    exit 1
fi
