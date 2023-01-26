CONDA_ENV=ml_pipeline
.PHONY: help

all: help

run: ## run the pipeline (train)
	python src/train.py \
		debug=false
debug: ## run the pipeline (train) with debugging enabled
	python src/train.py \
		debug=true

data: ## download the mnist data
	wget https://pjreddie.com/media/files/mnist_train.csv -O data/mnist_train.csv
	wget https://pjreddie.com/media/files/mnist_test.csv -O data/mnist_test.csv

install: environment.yml ## import any changes to env.yml into conda env
	conda env update -n ${CONDA_ENV} --file $^

env_export: ## export the conda envirnoment without package or name
	conda env export | head -n -1 | tail -n +2 > $@

help: ## display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

