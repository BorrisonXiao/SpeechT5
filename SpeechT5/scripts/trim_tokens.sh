#!/usr/bin/env bash

data_dir=/home/ec2-user/mult5/SpeechT5/data
text_dir=${data_dir}/text

# Get the current line count
total_lines=$(wc -l < ${text_dir}/text_train.token)
echo "Total lines: $total_lines"

# Calculate target lines for ~8GB (adjust ratio based on your file)
# If 13G is 100%, then 8G is about 61.5% of the lines
target_lines=$(echo "$total_lines * 45 / 100" | bc)  # 61% of original
echo "Target lines: $target_lines"

# Randomly sample lines using shuf (most efficient)
shuf ${text_dir}/text_train.token | head -n $target_lines > ${text_dir}/text_train_smaller.token

# Verify the new size
new_size=$(du -h ${text_dir}/text_train_smaller.token | cut -f1)
echo "New file size: $new_size"

# Replace the original (make backup first!)
cp ${text_dir}/text_train.token ${text_dir}/text_train.token.backup
mv ${text_dir}/text_train_smaller.token ${text_dir}/text_train.token