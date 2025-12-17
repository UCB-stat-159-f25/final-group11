# PHONY ALIASING
.PHONY: html, env

# Commands
env :
	conda env update -n loanstatus -f environment.yml 
clean:
	rm -f plots/*