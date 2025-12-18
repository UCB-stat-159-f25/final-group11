# PHONY ALIASING
.PHONY: env, clean, all

# Commands
env :
	conda env update -n loanstatus -f environment.yml 

all:
	conda run -n loanstatus jupyter nbconvert --execute --to notebook --inplace eda_data_cleaning.ipynb;
	conda run -n loanstatus jupyter nbconvert --execute --to notebook --inplace logistic_regression.ipynb;
	conda run -n loanstatus jupyter nbconvert --execute --to notebook --inplace knn.ipynb;

clean:
	rm -f plots/*