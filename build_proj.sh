pushd /workspace/fairseq

python setup.py clean --all
rm -rf build
rm fairseq/data/data_utils_fast.cpp
rm fairseq/data/token_block_utils_fast.cpp
python setup.py build_ext --inplace
pip install -e .

popd
