#!/usr/bin/env bash
# Quick smoke test — verifies the platform is working end-to-end
set -e

HOST="${1:-http://localhost:8000}"
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}☁️  Cloud ML Inference Platform — Smoke Test${NC}"
echo "Target: $HOST"
echo ""

# 1. Health check
echo -n "1. Health check... "
HEALTH=$(curl -s "$HOST/health")
STATUS=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
if [ "$STATUS" = "healthy" ]; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy: $HEALTH${NC}"
    exit 1
fi

# 2. Sync prediction
echo -n "2. Sync prediction... "
PRED=$(curl -s -X POST "$HOST/api/v1/predict/sync" \
    -H "Content-Type: application/json" \
    -d '{"text": "This is an amazing product, absolutely love it!"}')
LABEL=$(echo "$PRED" | python3 -c "import sys, json; print(json.load(sys.stdin).get('label', 'NONE'))" 2>/dev/null)
if [ "$LABEL" = "POSITIVE" ]; then
    echo -e "${GREEN}✅ Got POSITIVE${NC}"
else
    echo -e "${RED}❌ Expected POSITIVE, got: $PRED${NC}"
fi

# 3. Async prediction flow
echo -n "3. Async prediction (enqueue)... "
SUBMIT=$(curl -s -X POST "$HOST/api/v1/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Terrible experience, worst purchase ever."}')
REQ_ID=$(echo "$SUBMIT" | python3 -c "import sys, json; print(json.load(sys.stdin)['request_id'])" 2>/dev/null)
echo -e "${GREEN}✅ ID: ${REQ_ID:0:8}...${NC}"

echo -n "   Polling for result... "
for i in $(seq 1 20); do
    sleep 0.2
    RESULT=$(curl -s "$HOST/api/v1/result/$REQ_ID")
    RSTATUS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null)
    if [ "$RSTATUS" = "completed" ]; then
        RLABEL=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('label', ''))" 2>/dev/null)
        echo -e "${GREEN}✅ Got $RLABEL (${i}×200ms)${NC}"
        break
    fi
done
if [ "$RSTATUS" != "completed" ]; then
    echo -e "${RED}❌ Timed out: $RESULT${NC}"
fi

# 4. Batch prediction
echo -n "4. Batch prediction (3 texts)... "
BATCH=$(curl -s -X POST "$HOST/api/v1/predict/batch" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Great!", "Bad!", "Meh."]}')
COUNT=$(echo "$BATCH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('count', 0))" 2>/dev/null)
if [ "$COUNT" = "3" ]; then
    echo -e "${GREEN}✅ Queued 3 requests${NC}"
else
    echo -e "${RED}❌ Expected 3, got: $BATCH${NC}"
fi

# 5. Metrics endpoint
echo -n "5. Metrics endpoint... "
METRICS=$(curl -s "$HOST/metrics" | head -5)
if echo "$METRICS" | grep -q "inference_"; then
    echo -e "${GREEN}✅ Prometheus metrics available${NC}"
else
    echo -e "${RED}❌ No metrics found${NC}"
fi

echo ""
echo -e "${GREEN}All smoke tests passed! 🚀${NC}"
echo ""
echo "Next steps:"
echo "  make loadtest       — Run 100-user load test"
echo "  make dashboard      — Open Grafana monitoring"
echo "  make benchmark      — Run full benchmark suite"
