#!/bin/zsh
# launchd: Sundays at 08:00 AM AEST/AEDT

# set-up parameters
home=/Users/bryanpalmer
project=au-econ

# move to the project directory
cd $home/$project

# activate the uv environment
source $home/$project/.venv/bin/activate

# run the commodity and market chart modules
for run_set in energy yahoo asx; do
    python run.py "$run_set" \
        >>./LOGS/yahoo-commodities-log.log 2>>./LOGS/yahoo-commodities-err.log
done
