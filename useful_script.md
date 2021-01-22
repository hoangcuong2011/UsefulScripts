I found those scripts useful to preprocess a public wmt 2014 training dataset. They are inspired by this link: https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh



    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git

    SCRIPTS=mosesdecoder/scripts
    TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
    CLEAN=$SCRIPTS/training/clean-corpus-n.perl
    NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
    REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
    BPEROOT=subword-nmt/subword_nmt
    BPE_TOKENS=32000



    src=en
    tgt=fr
    prep=wmt14_en_fr
    mkdir $prep
    tmp=$prep/tmp
    orig=orig
    mkdir $prep/tmp
    mv en_all $prep/tmp/train.en 
    mv fr_all $prep/tmp/train.fr 

    echo "pre-processing train data..."
    cat $tmp/train.en | perl $NORM_PUNC en | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l en >> $tmp/train.tok.en
    cat $tmp/train.fr | perl $NORM_PUNC fr | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l fr >> $tmp/train.tok.fr


    cat $tmp/train.tok.en $tmp/train.tok.fr | subword-nmt learn-bpe -s BPE_TOKENS -o $tmp/wmt_en_fr_32K_bpe_code




    subword-nmt apply-bpe -c $tmp/wmt_en_fr_32K_bpe_code < $tmp/train.tok.fr | subword-nmt get-vocab > $tmp/train.tok.vocab.fr
    subword-nmt apply-bpe -c $tmp/wmt_en_fr_32K_bpe_code < $tmp/train.tok.en | subword-nmt get-vocab > $tmp/train.tok.vocab.en



    subword-nmt apply-bpe -c $tmp/wmt_en_fr_32K_bpe_code --vocabulary $tmp/train.tok.vocab.fr < $tmp/train.tok.fr > $tmp/train.tok.bpe.fr
    subword-nmt apply-bpe -c $tmp/wmt_en_fr_32K_bpe_code --vocabulary $tmp/train.tok.vocab.en < $tmp/train.tok.en > $tmp/train.tok.bpe.en


    perl $CLEAN -ratio 1.5 $tmp/train.tok.bpe $src $tgt $tmp/train.final.clean 1 150







Another example of such scripts




    SCRIPTS=/home/ubuntu/mosesdecoder/scripts
    TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
    CLEAN=$SCRIPTS/training/clean-corpus-n.perl
    NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
    REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

    cat ./train.en | perl $NORM_PUNC en | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l en >> ./train.tok.en



    SCRIPTS=/home/ubuntu/mosesdecoder/scripts
    TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
    CLEAN=$SCRIPTS/training/clean-corpus-n.perl
    NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
    REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

    cat ./train.fr | perl $NORM_PUNC fr | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l fr >> ./train.tok.fr


    cat ./train.tok.en ./train.tok.fr | subword-nmt learn-bpe -s 32000 -o ./wmt_en_fr_32K_bpe_code



    nohup subword-nmt apply-bpe -c ./wmt_en_fr_32K_bpe_code < ./train.tok.fr > ./train.tok.bpe.fr &
    nohup subword-nmt apply-bpe -c ./wmt_en_fr_32K_bpe_code < ./train.tok.en > ./train.tok.bpe.en &
