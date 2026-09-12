.PHONY: all install data warehouse analysis dashboard test clean

all: data warehouse analysis

install:
	pip install -r requirements.txt

data:            ## generate the simulated network (~30s, ~7.4M transactions)
	python src/generate_data.py

warehouse:       ## run sql/*.sql in order into data/warehouse.duckdb
	python src/warehouse.py

analysis:        ## metric evaluation, drivers, segmentation, forecast
	cd src && python evaluate_churn_definition.py
	cd src && python churn_drivers.py
	cd src && python segmentation.py
	cd src && python forecast.py
	cd src && python export_for_bi.py

dashboard:       ## serve the Streamlit demo (reads reports/, no pipeline run needed)
	streamlit run app.py

test:
	pytest -q tests

clean:
	rm -rf data/raw data/warehouse.duckdb reports/figures/*.png
