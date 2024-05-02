#check if one command line argument is provided else throw an error
if [ $# -ne 1 ]; then
  echo "Usage: $0 <path to conda>"
  exit 1
fi

conda create -n $1 python=3.12 -y -q 1>/dev/null
source activate $1

# Install dependencies
python -m pip install cmake -q 1>/dev/null
python -m pip install PQAnalysis -q 1>/dev/null

rm -rf PQ

git clone https://github.com/MolarVerse/PQ.git 1>/dev/null
cd PQ
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 1>/dev/null
make 1>/dev/null
make install 1>/dev/null
cd ..
cd ..

# Deactivate environment
source deactivate 1>/dev/null
