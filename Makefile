CONDA_ENV=ml_pipeline

all: run

run:
	./launch.sh

data:
	python src/data.py

batch:
	python src/batch.py

install:
	conda env updates -n ${CONDA_ENV} --file environment.yml
