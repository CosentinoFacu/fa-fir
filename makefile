# Nombre del archivo Python a ejecutar
SCRIPT = scripts/script_to_test.py

# make para correr el script to test
run_sc:
	. venv/bin/activate && \
	python3 $(SCRIPT)

setup_environment:
	python3 -m virtualenv venv && \
	. venv/bin/activate && \
	python3 -m pip install -r requirements.txt