normalize:
	python train.py > log.log &

generate_pdf:
	markdown-pdf README.md
	mv README.pdf report.pdf