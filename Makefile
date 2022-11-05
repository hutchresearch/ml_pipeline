CONDA_ENV=ml_pipeline

all: run

run:
	python src/pipeline.py train

data:
	python src/data.py

batch:
	python src/batch.py

install:
	conda env updates -n ${CONDA_ENV} --file environment.yml
