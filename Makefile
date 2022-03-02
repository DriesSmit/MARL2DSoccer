run:
	./football_env/bin/python3 ./main.py
build_env:
	python3 -m venv football_env
	./football_env/bin/pip3 install -r requirements.txt

