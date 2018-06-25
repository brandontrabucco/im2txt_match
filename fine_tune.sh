python2 $SCRATCH/research/repos/im2txt_match/im2txt_match/train.py \
--input_file_pattern="$SCRATCH/research/data/coco/train-?????-of-00256" \
--train_dir="$SCRATCH/research/ckpts/im2txt_match/train/" \
--train_inception=true \
--number_of_steps=300000 \
