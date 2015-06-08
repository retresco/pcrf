rem Annotate data
..\bin\ner-annotate -c chunk.cfg ..\data\chunker-train.corpus > chunker-train.annotated-corpus
rem Train model
..\bin\crf-train -m chunker.model -n 50 --order 1 chunker-train.annotated-corpus
rem Evaluate model
..\bin\crf-apply --eval -c chunk.cfg -m chunker.model --order 1 ..\data\chunker-test.corpus > chunker.result
