wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p ~/miniconda3
rm ~/miniconda.sh
echo "export PATH=$HOME/miniconda3/bin:$PATH" > $HOME/.profile
