#check if one command line argument is provided else throw an error
if [ $# -ne 1 ]; then
  echo "Usage: $0 <name of env>"
  exit 1
fi

# print nice message to the user
echo "#################################################"
echo "#       Building PQAnalysis conda package       #"
echo "#################################################"
echo ""

echo "(!) creating conda environment $1 with python 3.12"
echo ""
conda create -n $1 python=3.12 -y -q 1>/dev/null

echo "(!) activating conda environment $1"
echo ""
source activate $1

# Install dependencies
echo "(!) installing dependencies"
echo ""
python -m pip install cmake -q 1>/dev/null

rm -rf PQ

# check if gcc version is at least 13
gcc_version=$(gcc --version | grep ^gcc | sed 's/^.* //g')
if [ $(echo "$gcc_version >= 13" | bc) -eq 0 ]; then
  echo "Error: gcc version must be at least 13"
  exit 1
fi

# Clone and build PQ
echo "(!) cloning PQ from github"
echo ""
git clone https://github.com/MolarVerse/PQ.git -q 1>/dev/null
cd PQ
mkdir build
cd build
# Build PQ
echo "(!) building PQ"
echo ""
#write cmake to /dev/null stderr and stdout and check if it is successful
cmake_success=$(cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 2>&1 >/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: cmake failed"
  echo $cmake_success
  exit 1
fi
#write make to /dev/null stderr and stdout and check if it is successful
make_success=$(make 2>&1 >/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: make failed"
  echo $make_success
  exit 1
fi
#write make install to /dev/null stderr and stdout and check if it is successful
make_install_success=$(make install 2>&1 >/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: make install failed"
  echo $make_install_success
  exit 1
fi
cd ..
cd ..

echo "#################################################"
echo "#          conda env setup finished             #"
echo "#################################################"
echo "#                                               #"
echo "#  To activate the environment run:             #"

#adjust length of whitespaces after $1 before #
# calculate length of $1
length=$(echo -n $1 | wc -c)
# calculate length of "source activate "
length2=$(echo -n "source activate " | wc -c)
# calculate total length of string
total_length=$(echo -n "#################################################" | wc -c)
# calculate number of whitespaces
whitespaces=$((total_length - length - length2 - 5))
# if whitespaces is negative set it to 1
if [ $whitespaces -lt 1 ]; then
  whitespaces=0
fi

echo "#  source activate $1$(printf '%*s' $whitespaces) #"
echo "#                                               #"
echo "#  To deactivate the environment run:           #"
echo "#  source deactivate                            #"
echo "#################################################"
