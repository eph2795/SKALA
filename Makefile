lint:
	black . -l 120
	ruff check --fix
deps:
	sed -i 's/>/~/g' pyproject.toml
use1:
	uv run -m use_scripts.batch_analysis_report
use2:
	uv run -m use_scripts.classify_and_plot
use3:
	uv run -m use_scripts.cpu_inference
use4: 
	uv run -m use_scripts.synthetic_data_gen
