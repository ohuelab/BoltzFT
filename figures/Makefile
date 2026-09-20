SCRIPTS := \
	make_learning_curve.py \
	make_absolute_metrics.py \
	make_cascade.py \
	make_supervised_baselines.py

.PHONY: figures clean

figures:
	@for s in $(SCRIPTS); do \
		echo "[figuregen] $$s"; \
		uv run --locked --project . python $$s || exit 1; \
	done

clean:
	rm -rf __pycache__
