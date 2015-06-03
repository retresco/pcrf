input=guardian.text
traincorpus=../data/chunker-train.corpus
testcorpus=../data/chunker-test.corpus


# Annotate data
echo "================================================================"
echo "Annotating " $traincorpus
echo "================================================================"
../ner-annotate -c chunk.cfg $traincorpus > $traincorpus.annotated

# Train model
echo
echo "================================================================"
echo "Training " $traincorpus
echo "================================================================"
../crf-train -m chunker.model -n 50 --order 1 $traincorpus.annotated

# Evaluate model
echo
echo "================================================================"
echo "Evaluating " $testcorpus
echo "================================================================"
../crf-apply --eval -c chunk.cfg -m chunker.model --order 1 $testcorpus > chunker.result

echo
echo "================================================================"
echo "Apply CRF-model to running text in " $input
echo "================================================================"
# Apply model to running text
../crf-apply -c chunk.cfg --running-text -m chunker.model --order 1 $input > $input.result


