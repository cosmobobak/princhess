#!/bin/bash

set -e

PRINCHESS=${PRINCHESS:-../target/release/princhess}

generate_data() {
  echo "Sampling data..."

  ls pgn/*.pgn | parallel -u -j 9 $PRINCHESS -t {} -o {}.libsvm

  rm -f model_data/*.libsvm.*

  for f in pgn/*.pgn
  do
    echo "Calculating split..."

    samples=$(wc -l < $f.libsvm)
    splits=$(( $samples / 1000000 ))
    split_size=$(( $samples / $splits + 1))

    echo "Splitting data ($split_size)..."

    split -l $split_size $f.libsvm model_data/$(basename $f).libsvm.

    rm $f.libsvm
    rm -f model_data/*.gz
  done
}

generate_data
