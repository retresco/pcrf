input=guardian.text
traincorpus=train.txt
testcorpus=test.txt

echo "================================================================"
echo "Ensuring chunking data from CoNLL2000 Shared Task"
echo "================================================================"
if [ ! -f "$traincorpus" ]
then
    wget -nc http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
    gunzip train.txt.gz
fi

if [ ! -f "$testcorpus" ]
then
    wget -nc http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
    gunzip test.txt.gz
fi

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
echo "Apply CRF-model to running text in " $input ","
echo "creating " $input.result
echo "================================================================"
# Apply model to running text
../crf-apply -c chunk.cfg --running-text -m chunker.model --order 1 $input > $input.result


