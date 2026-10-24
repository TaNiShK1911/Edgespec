# EdgeSpec-H2O Makefile

.PHONY: check test gate1 gate2 gate3 lint clean logs

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

# Gate runs
gate1:
	bash bench/gate1.sh

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
