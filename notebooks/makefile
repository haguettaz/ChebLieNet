NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	jupyter nbconvert --clear-output $(NB)
	rm -rf .ipynb_checkpoints
	rm -rf saved_graphs

.PHONY: run $(NB) clean
