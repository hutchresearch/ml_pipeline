PYTHON_VERSION=3.10
ENV_NAME=ml_pipeline
INSTALL_DIR=$HOME/Dev
# for wwu research:
# INSTALL_DIR=/research/hutchinson/workspace/$USERNAME

####################
#
# download miniconda
#
####################
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/Downloads/Miniconda3-latest-Linux-x86_64.sh

####################
#
# run install script
# headless
#
####################
rm -rf $INSTALL_DIR/miniconda3
bash $HOME/Downloads/Miniconda3-latest-Linux-x86_64.sh -b -p $INSTALL_DIR/miniconda3

####################
#
# create first conda environment
#
####################
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

################
#
# place the following in $HOME/.bashrc
#
# then use `hutchconda` to activate base env
#
################

# WORKSPACE_DIR=/research/hutchinson/workspace/$USERNAME
# hutchconda() {
#     __conda_setup="$('$WORKSPACE_DIR/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#     if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
#     else
#     if [ -f "$WORKSPACE_DIR/miniconda3/etc/profile.d/conda.sh" ]; then
#     . "$WORKSPACE_DIR/miniconda3/etc/profile.d/conda.sh"
#     else
#     export PATH="$WORKSPACE_DIR/miniconda3/bin:$PATH"
#     fi
#     fi
#     unset __conda_setup
# }


####################
#
# activate conda environment
#
####################
conda activate $ENV_NAME

####################
#
# install pytorch
#
####################
conda install -c pytorch pytorch -y

####################
#
# or install from envirnoment.yml
#
####################
conda env update -n $ENV_NAME --file environment.yml
