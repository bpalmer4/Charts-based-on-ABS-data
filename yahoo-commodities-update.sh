#!/bin/zsh
# launchd: Saturdays at 10:30 AM AEST/AEDT

# set-up parameters
home=/Users/bryanpalmer
project=ABS

# move to the project directory
cd $home/$project

# activate the uv environment
source $home/$project/.venv/bin/activate

# run the Yahoo commodities notebook
jupyter-nbconvert --to notebook --execute --inplace \
    ./notebooks/YAHOO_daily_commodities.ipynb \
    >>./LOGS/yahoo-commodities-log.log 2>>./LOGS/yahoo-commodities-err.log
