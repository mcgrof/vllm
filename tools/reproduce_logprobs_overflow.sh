#!/bin/bash
# Reproducer: LogprobsTensors OverflowError with CartridgeConnector
#
# Real-world scenario:
#   A user trains a cartridge (pre-computed KV cache) for a document
#   and wants to serve it with vLLM. They request prompt_logprobs
#   (e.g. for MC scoring, perplexity measurement, or evaluation) and
#   the server crashes with OverflowError.
#
# Root cause:
#   vllm/v1/outputs.py LogprobsTensors.empty_cpu() uses torch.empty()
#   which leaves token IDs uninitialized. Positions that the KV
#   connector marks as "externally computed" are never filled by the
#   model runner, so they contain garbage int32 values that crash
#   tokenizer.decode().
#
# Prerequisites:
#   - GPU (any: CUDA or ROCm)
#   - vLLM installed from 20260430-cartridges-upstream branch
#   - HF token for Llama-3.2-3B-Instruct
#
# To demonstrate the bug, run against the UNFIXED code:
#   git checkout 20260429-cartridges-code-only~2  # before LogprobsTensors fix
#
# To verify the fix works:
#   git checkout 20260430-cartridges-upstream      # has the fix
#
# Usage:
#   bash tools/reproduce_logprobs_overflow.sh

set -e

MODEL="meta-llama/Llama-3.2-3B-Instruct"
PORT=8199
CART_PATH="/tmp/reproduce_cart.pt"

echo "=== Step 1: Build a cartridge from a known document ==="
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

tok = AutoTokenizer.from_pretrained('$MODEL')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL', torch_dtype=torch.bfloat16).to('cuda').eval()

doc = ('The patient John Smith was admitted on March 15 2026 with '
       'chest pain. Troponin-I was 2.4 ng/mL. Blood pressure 145/92. '
       'Heart rate 88 bpm. History of type 2 diabetes since 2018.')
ids = tok.encode(doc, add_special_tokens=False)
cache = DynamicCache()
with torch.no_grad():
    model(input_ids=torch.tensor([ids], device='cuda'),
          past_key_values=cache, use_cache=True, return_dict=True)

Ks, Vs = [], []
for li in range(len(cache.layers)):
    layer = cache.layers[li]
    k = layer.keys.cpu() if hasattr(layer, 'keys') else layer.key_cache[0].cpu()
    v = layer.values.cpu() if hasattr(layer, 'values') else layer.value_cache[0].cpu()
    Ks.append(k); Vs.append(v)
torch.save({'trainable_keys': Ks, 'trainable_values': Vs,
            'frozen_keys': [], 'frozen_values': []}, '$CART_PATH')
print('Cartridge saved: %d layers, %d tokens' % (len(Ks), Ks[0].shape[-2]))
del model; torch.cuda.empty_cache()
"

echo ""
echo "=== Step 2: Start vLLM with CartridgeConnector ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --kv-transfer-config "{\"kv_connector\":\"CartridgeConnector\",\"kv_connector_module_path\":\"vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector\",\"kv_connector_extra_config\":{\"cartridge_path\":\"$CART_PATH\"},\"kv_role\":\"kv_both\"}" \
    &
SERVER_PID=$!

# Wait for startup
for i in $(seq 1 30); do
    curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
    sleep 5
done

if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "FAIL: server did not start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "Server ready (PID $SERVER_PID)"

echo ""
echo "=== Step 3: Send request with logprobs (triggers the bug) ==="
echo ""
echo "Sending: /v1/completions with logprobs=1 + echo=true"
echo "On UNFIXED code, this crashes with OverflowError."
echo "On FIXED code, this returns valid logprobs."
echo ""

PROMPT="The patient John Smith was admitted on March 15 2026 with chest pain. Troponin-I was 2.4 ng/mL. Blood pressure 145/92. Heart rate 88 bpm. History of type 2 diabetes since 2018. What was the troponin level?"

RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":5,\"temperature\":0,\"logprobs\":1,\"echo\":true}")

if echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d:
    print('BUG TRIGGERED: %s' % d['error']['message'][:200])
    sys.exit(1)
lps = d['choices'][0].get('logprobs', {}).get('token_logprobs', [])
print('FIX WORKS: got %d logprobs, server did not crash' % len(lps))
" 2>/dev/null; then
    echo ""
    echo "=== RESULT: Fix is in place. logprobs returned without crash. ==="
else
    echo ""
    echo "=== RESULT: BUG REPRODUCED. Server crashed on prompt_logprobs ==="
    echo "            with externally-computed CartridgeConnector tokens. ==="
fi

echo ""
echo "=== Cleanup ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
rm -f "$CART_PATH"
echo "Done."
