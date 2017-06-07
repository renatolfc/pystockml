all: report.pdf

report.pdf: report.md mytemplate.tex
	pandoc --toc report.md -o report.pdf \
		--template=mytemplate.tex \
		--variable papersize=a4
