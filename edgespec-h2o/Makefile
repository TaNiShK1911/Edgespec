.PHONY: check test gate1 gate1-laptop gate1-fast gate1-full gate2 gate2-laptop gate2-full gate3 lint clean logs

# Environment check
check:
	python tools/check_env.py

# Run all tests
test:
	python -m pytest tests/ -v --tb=short

# Run individual test suites
test-tokenizer:
	python -m pytest tests/test_tokenizer.py -v

test-qlinear:
	python -m pytest tests/test_qlinear.py -v

test-model:
	python -m pytest tests/test_model.py -v

test-h2o:
	python -m pytest tests/test_h2o_cache.py -v

# Gate runs
gate1:
	bash bench/gate1.sh

# Gate 1 — laptop mode (no weights, runs only code/tokenizer/quant checks)
gate1-laptop:
	bash bench/gate1.sh --no-weights --skip-thermal --skip-kernel

# Gate 1 — fast Nano run (skip 10-min thermal soak)
gate1-fast:
	bash bench/gate1.sh \
		--weights data/weights/opt-125m-4bit.edgespec \
		--model-size 125m \
		--skip-thermal

# Gate 1 — full Nano run (all checks including thermal soak)
gate1-full:
	bash bench/gate1.sh \
		--weights data/weights/opt-1.3b-4bit.edgespec \
		--model-size 1.3b \
		--decode-tps-min 8 \
		--mem-budget-mb 3800

# Gate 2 — H2O KV cache
gate2:
	bash bench/gate2.sh

# Gate 2 — laptop mode (no weights, code checks only)
gate2-laptop:
	bash bench/gate2.sh --no-weights --skip-soak --device cpu

# Gate 2 — full Nano run
gate2-full:
	bash bench/gate2.sh \
		--weights data/weights/opt-1.3b-4bit.edgespec \
		--model-size 1.3b

# Offline quantization (laptop only)
quantize-125m:
	python tools/offline_quantize.py --model facebook/opt-125m

quantize-1.3b:
	python tools/offline_quantize.py --model facebook/opt-1.3b

# Export golden files (laptop only)
golden-125m:
	python tools/export_golden.py --model facebook/opt-125m

golden-1.3b:
	python tools/export_golden.py --model facebook/opt-1.3b

# Generate synthetic logs
logs:
	python tools/gen_logs.py --n 1000 --format both

# PPL evaluation
eval-ppl:
	python tools/eval_ppl.py --model facebook/opt-125m

# Lint (Python 3.6 compile check)
lint:
	find edgespec/ -name '*.py' | xargs python -m py_compile
	find tools/ -name '*.py' | xargs python -m py_compile
	find tests/ -name '*.py' | xargs python -m py_compile

# Nano setup
nano-setup:
	sudo nvpmodel -m 0 && sudo jetson_clocks && sudo nvpmodel -q
	tegrastats --interval 500 --logfile results/tegrastats.log &

# Clean
clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf results/*.json results/*.log results/*.csv

