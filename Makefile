# PHONY ALIASING
.PHONY: env, clean, all

# Commands
env :
	conda env update -n loanstatus -f environment.yml 

all:
	jupyter nbconvert --to notebook --execute Part1.ipynb --output Part1.ipynb
	jupyter nbconvert --to notebook --execute Part2.ipynb --output Part2.ipynb
	jupyter nbconvert --to notebook --execute Part3.ipynb --output Part3.ipynb

clean:
	rm -f plots/*