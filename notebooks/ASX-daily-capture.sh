#!/bin/zsh
# crontab: 01 20 * * 1-5 /Users/bryanpalmer/ABS/notebooks/ASX-daily-capture.sh
# note: * * * * * ==> minute hour day-of-month month day-of-week

# set-up parameters
home=/Users/bryanpalmer
project=ABS
working=notebooks
runrun=asx_daily_data_capture.py

# move to the home directory
cd $home

# activate the uv environment
source $home/$project/.venv/bin/activate

# move to the working directory
cd $project/$working

# initiate the data capture
python ./$runrun >>./LOGS/asx-log.log 2>>./LOGS/asx-err.log

